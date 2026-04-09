"""
Backup/restore ChromaDB analyses to Azure Blob Storage for persistence.
"""

import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential
import structlog

logger = structlog.get_logger()


class BlobBackupManager:
    """Manages backup and restore of analysis reports to Azure Blob Storage."""

    def __init__(self, storage_account_name: Optional[str] = None):
        """
        Initialize blob backup manager.

        Args:
            storage_account_name: Azure Storage account name. If None, reads from env.
        """
        self.storage_account_name = storage_account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

        if not self.storage_account_name:
            logger.warning("No storage account configured - blob backup disabled")
            self.enabled = False
            return

        self.enabled = True
        self.container_name = "stocksquad-analyses"

        # Use managed identity or connection string
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use managed identity
            account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
            credential = DefaultAzureCredential()
            self.blob_service_client = BlobServiceClient(account_url, credential=credential)

        # Ensure container exists
        try:
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            if not self.container_client.exists():
                self.container_client.create_container()
                logger.info(f"Created blob container: {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to initialize blob storage: {e}")
            self.enabled = False

    def save_analysis(self, doc_id: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Save an analysis to blob storage.

        Args:
            doc_id: Document ID (e.g., "AAPL_2024-01-15_12-30-45")
            analysis_data: Full analysis data dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            blob_name = f"analyses/{doc_id}.json"
            blob_client = self.container_client.get_blob_client(blob_name)

            # Convert to JSON and upload
            json_data = json.dumps(analysis_data, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)

            logger.info(f"Saved analysis to blob storage: {blob_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save analysis {doc_id} to blob storage: {e}")
            return False

    def load_analysis(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an analysis from blob storage.

        Args:
            doc_id: Document ID

        Returns:
            Analysis data dictionary or None if not found
        """
        if not self.enabled:
            return None

        try:
            blob_name = f"analyses/{doc_id}.json"
            blob_client = self.container_client.get_blob_client(blob_name)

            if not blob_client.exists():
                return None

            # Download and parse JSON
            blob_data = blob_client.download_blob().readall()
            analysis_data = json.loads(blob_data)

            logger.info(f"Loaded analysis from blob storage: {blob_name}")
            return analysis_data

        except Exception as e:
            logger.error(f"Failed to load analysis {doc_id} from blob storage: {e}")
            return None

    def list_analyses(self) -> List[str]:
        """
        List all analysis document IDs in blob storage.

        Returns:
            List of document IDs
        """
        if not self.enabled:
            return []

        try:
            blobs = self.container_client.list_blobs(name_starts_with="analyses/")
            doc_ids = [
                blob.name.replace("analyses/", "").replace(".json", "")
                for blob in blobs
            ]
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to list analyses from blob storage: {e}")
            return []

    def restore_all_to_memory(self, memory_instance) -> int:
        """
        Restore all analyses from blob storage to ChromaDB.

        Args:
            memory_instance: LongTermMemory instance

        Returns:
            Number of analyses restored
        """
        if not self.enabled:
            logger.info("Blob backup not enabled - skipping restore")
            return 0

        try:
            doc_ids = self.list_analyses()
            restored_count = 0
            skipped_count = 0

            # Get existing doc IDs from ChromaDB to avoid duplicates
            try:
                existing = memory_instance.collection.get()
                existing_ids = set(existing['ids']) if existing and 'ids' in existing else set()
                logger.info(f"Found {len(existing_ids)} existing analyses in ChromaDB")
            except Exception as e:
                logger.warning(f"Failed to check existing analyses: {e}")
                existing_ids = set()

            for doc_id in doc_ids:
                # Skip if already exists in ChromaDB
                if doc_id in existing_ids:
                    skipped_count += 1
                    continue

                analysis_data = self.load_analysis(doc_id)
                if analysis_data:
                    # Store in ChromaDB with original doc_id
                    try:
                        memory_instance._restore_analysis_with_id(
                            doc_id=doc_id,
                            ticker=analysis_data.get("ticker"),
                            analysis_summary=analysis_data.get("summary"),
                            full_analysis=analysis_data.get("full_analysis"),
                            metadata=analysis_data.get("metadata"),
                        )
                        restored_count += 1
                    except Exception as e:
                        logger.error(f"Failed to restore {doc_id} to ChromaDB: {e}")

            logger.info(f"Restored {restored_count} new analyses, skipped {skipped_count} existing (total {len(doc_ids)} in blob storage)")
            return restored_count

        except Exception as e:
            logger.error(f"Failed to restore analyses: {e}")
            return 0
