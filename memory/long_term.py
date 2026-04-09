"""
Long-term memory implementation using ChromaDB vector store.
Stores and retrieves past analyses by ticker with semantic search capabilities.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import pandas as pd
import numpy as np

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.blob_backup import BlobBackupManager


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle pandas/numpy types."""

    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


class LongTermMemory:
    """
    Long-term memory manager using ChromaDB for persistent storage.

    Stores:
    - Past analysis results indexed by ticker + date
    - Embeddings for semantic search
    - Agent findings and recommendations
    """

    def __init__(self):
        """Initialize long-term memory with ChromaDB and Azure OpenAI embeddings."""
        print("  [1/4] Loading settings...")
        self.settings = get_settings()

        # Initialize ChromaDB client
        print(f"  [2/4] Initializing ChromaDB at {self.settings.chroma_db_path}...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection for stock analyses
        print("  [3/4] Getting/creating collection 'stock_analyses'...")
        self.collection = self.chroma_client.get_or_create_collection(
            name="stock_analyses",
            metadata={"description": "Stock analysis results from StockSquad"},
        )

        # Initialize Azure OpenAI client for embeddings
        # Use API key if provided, otherwise fall back to managed identity
        print("  [4/4] Initializing Azure OpenAI client...")
        if self.settings.azure_openai_api_key:
            print("       → Using API key authentication")
            self.openai_client = AzureOpenAI(
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
            )
        else:
            # Use managed identity when no API key is provided
            print("       → Using managed identity authentication")
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.openai_client = AzureOpenAI(
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                azure_ad_token_provider=token_provider,
            )

        print("  ✅ LongTermMemory initialized successfully!")

        # Initialize blob backup for persistence
        print("  [5/5] Initializing blob backup...")
        self.blob_backup = BlobBackupManager()
        if self.blob_backup.enabled:
            print("       → Blob backup enabled, restoring analyses...")
            restored = self.blob_backup.restore_all_to_memory(self)
            print(f"       → Restored {restored} analyses from blob storage")
        else:
            print("       → Blob backup disabled (no storage account configured)")

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Azure OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.settings.azure_openai_embedding_deployment_name,
        )
        return response.data[0].embedding

    def _create_document_id(self, ticker: str, timestamp: datetime) -> str:
        """
        Create a unique document ID for an analysis.

        Args:
            ticker: Stock ticker
            timestamp: Analysis timestamp

        Returns:
            Unique document ID
        """
        return f"{ticker.upper()}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def store_analysis(
        self,
        ticker: str,
        analysis_summary: str,
        full_analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store an analysis in long-term memory.

        Args:
            ticker: Stock ticker
            analysis_summary: Text summary for embedding/search
            full_analysis: Complete analysis data
            metadata: Optional additional metadata

        Returns:
            Document ID of stored analysis
        """
        timestamp = datetime.now()
        doc_id = self._create_document_id(ticker, timestamp)

        # Generate embedding for the summary
        embedding = self._generate_embedding(analysis_summary)

        # Prepare metadata
        doc_metadata = {
            "ticker": ticker.upper(),
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            **(metadata or {}),
        }

        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[analysis_summary],
            metadatas=[doc_metadata],
        )

        # Store full analysis as JSON in a separate file (local)
        analysis_data = {
            "ticker": ticker.upper(),
            "timestamp": timestamp.isoformat(),
            "summary": analysis_summary,
            "full_analysis": full_analysis,
            "metadata": doc_metadata,
        }

        analysis_path = self.settings.chroma_db_path / "analyses"
        analysis_path.mkdir(exist_ok=True)
        with open(analysis_path / f"{doc_id}.json", "w") as f:
            json.dump(analysis_data, f, indent=2, cls=CustomJSONEncoder)

        # Backup to blob storage for persistence
        if self.blob_backup.enabled:
            self.blob_backup.save_analysis(doc_id, analysis_data)

        return doc_id

    def _restore_analysis_with_id(
        self,
        doc_id: str,
        ticker: str,
        analysis_summary: str,
        full_analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Restore an analysis with a specific document ID (used when restoring from backup).

        Args:
            doc_id: Existing document ID to use
            ticker: Stock ticker
            analysis_summary: Text summary for embedding/search
            full_analysis: Complete analysis data
            metadata: Optional additional metadata

        Returns:
            Document ID of stored analysis
        """
        # Generate embedding for the summary
        embedding = self._generate_embedding(analysis_summary)

        # Use provided metadata or extract from analysis
        doc_metadata = metadata or {
            "ticker": ticker.upper(),
            "timestamp": full_analysis.get("timestamp", datetime.now().isoformat()),
            "date": full_analysis.get("timestamp", datetime.now().isoformat())[:10],
        }

        # Store in ChromaDB with original doc_id
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[analysis_summary],
            metadatas=[doc_metadata],
        )

        # Store full analysis as JSON in a separate file (local)
        analysis_data = {
            "ticker": ticker.upper(),
            "timestamp": doc_metadata.get("timestamp"),
            "summary": analysis_summary,
            "full_analysis": full_analysis,
            "metadata": doc_metadata,
        }

        analysis_path = self.settings.chroma_db_path / "analyses"
        analysis_path.mkdir(exist_ok=True)
        with open(analysis_path / f"{doc_id}.json", "w") as f:
            json.dump(analysis_data, f, indent=2, cls=CustomJSONEncoder)

        return doc_id

    def retrieve_past_analyses(
        self,
        ticker: str,
        limit: int = 5,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve past analyses for a ticker.

        Args:
            ticker: Stock ticker
            limit: Maximum number of analyses to retrieve
            days_back: Only retrieve analyses from last N days

        Returns:
            List of past analyses
        """
        # Query ChromaDB for this ticker
        where_filter = {"ticker": ticker.upper()}

        results = self.collection.get(
            where=where_filter,
            limit=limit,
        )

        if not results["ids"]:
            return []

        # Load full analyses from JSON files
        analyses = []
        analysis_path = self.settings.chroma_db_path / "analyses"

        for doc_id, summary, metadata in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            try:
                # Load full analysis if available
                full_path = analysis_path / f"{doc_id}.json"
                if full_path.exists():
                    with open(full_path, "r") as f:
                        full_data = json.load(f)
                    analyses.append(full_data)
                else:
                    # Fallback to just metadata and summary
                    analyses.append(
                        {
                            "ticker": ticker.upper(),
                            "summary": summary,
                            "metadata": metadata,
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not load analysis {doc_id}: {e}")
                continue

        # Sort by timestamp (most recent first)
        analyses.sort(
            key=lambda x: x.get("timestamp", ""), reverse=True
        )

        return analyses[:limit]

    def delete_analysis(self, doc_id: str) -> bool:
        """
        Delete an analysis from all storage locations.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        success = True

        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[doc_id])
            print(f"Deleted {doc_id} from ChromaDB")
        except Exception as e:
            print(f"Failed to delete {doc_id} from ChromaDB: {e}")
            success = False

        try:
            # Delete local JSON file
            analysis_path = self.settings.chroma_db_path / "analyses" / f"{doc_id}.json"
            if analysis_path.exists():
                analysis_path.unlink()
                print(f"Deleted {doc_id} local file")
        except Exception as e:
            print(f"Failed to delete {doc_id} local file: {e}")
            success = False

        # Delete from blob storage if enabled
        try:
            blob_deleted = self.blob_backup.delete_analysis(doc_id)
            if blob_deleted:
                print(f"Deleted {doc_id} from blob storage")
            elif self.blob_backup.enabled:
                print(f"Warning: {doc_id} not found in blob storage")
        except Exception as e:
            print(f"Failed to delete {doc_id} from blob storage: {e}")
            success = False

        return success

    def semantic_search(
        self, query: str, ticker: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across analyses.

        Args:
            query: Search query
            ticker: Optional ticker filter
            limit: Maximum results

        Returns:
            List of matching analyses with similarity scores
        """
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)

        # Build filter
        where_filter = {"ticker": ticker.upper()} if ticker else None

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=limit,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Load full analyses
        analyses = []
        analysis_path = self.settings.chroma_db_path / "analyses"

        for doc_id, distance, summary, metadata in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            try:
                full_path = analysis_path / f"{doc_id}.json"
                if full_path.exists():
                    with open(full_path, "r") as f:
                        full_data = json.load(f)
                    full_data["similarity_score"] = 1 - distance  # Convert distance to similarity
                    analyses.append(full_data)
            except Exception as e:
                print(f"Warning: Could not load analysis {doc_id}: {e}")
                continue

        return analyses

    def get_analysis_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific analysis by document ID.

        Args:
            doc_id: Document ID

        Returns:
            Analysis data or None if not found
        """
        analysis_path = self.settings.chroma_db_path / "analyses"
        full_path = analysis_path / f"{doc_id}.json"

        if not full_path.exists():
            return None

        try:
            with open(full_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading analysis {doc_id}: {e}")
            return None

    def delete_analysis(self, doc_id: str) -> bool:
        """
        Delete an analysis from long-term memory.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[doc_id])

            # Delete JSON file
            analysis_path = self.settings.chroma_db_path / "analyses"
            full_path = analysis_path / f"{doc_id}.json"
            if full_path.exists():
                full_path.unlink()

            return True
        except Exception as e:
            print(f"Error deleting analysis {doc_id}: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored analyses.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "total_analyses": count,
            "collection_name": self.collection.name,
            "storage_path": str(self.settings.chroma_db_path),
        }
