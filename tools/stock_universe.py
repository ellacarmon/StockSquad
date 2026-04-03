"""
Stock Universe Management
Manages lists of stocks for screening and analysis.
"""

import json
import os
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime


class StockUniverse:
    """Manages stock universes for screening."""

    def __init__(self):
        """Initialize stock universe manager."""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.universes = {}
        self._load_universes()

    def _load_universes(self):
        """Load all available universes from data directory."""
        if not self.data_dir.exists():
            print(f"Warning: Data directory not found: {self.data_dir}")
            return

        for file in self.data_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    universe_name = file.stem
                    self.universes[universe_name] = data
            except Exception as e:
                print(f"Warning: Failed to load universe {file}: {e}")

    def get_universe(self, name: str = "sp100") -> Dict:
        """
        Get a stock universe by name.

        Args:
            name: Universe name (sp100, sp500, nasdaq100, etc.)

        Returns:
            Universe dictionary with tickers and metadata
        """
        if name not in self.universes:
            raise ValueError(f"Universe '{name}' not found. Available: {list(self.universes.keys())}")

        return self.universes[name]

    def get_tickers(self, universe: str = "sp100") -> List[str]:
        """
        Get list of tickers from a universe.

        Args:
            universe: Universe name

        Returns:
            List of ticker symbols
        """
        universe_data = self.get_universe(universe)
        return universe_data.get("tickers", [])

    def get_sectors(self, universe: str = "sp100") -> Dict[str, List[str]]:
        """
        Get sector mappings for a universe.

        Args:
            universe: Universe name

        Returns:
            Dictionary mapping sectors to tickers
        """
        universe_data = self.get_universe(universe)
        return universe_data.get("sectors", {})

    def get_tickers_by_sector(self, sector: str, universe: str = "sp100") -> List[str]:
        """
        Get tickers for a specific sector.

        Args:
            sector: Sector name (Technology, Healthcare, etc.)
            universe: Universe name

        Returns:
            List of tickers in that sector
        """
        sectors = self.get_sectors(universe)

        # Case-insensitive sector matching
        for sector_name, tickers in sectors.items():
            if sector_name.lower() == sector.lower():
                return tickers

        return []

    def list_universes(self) -> List[str]:
        """
        List all available universes.

        Returns:
            List of universe names
        """
        return list(self.universes.keys())

    def list_sectors(self, universe: str = "sp100") -> List[str]:
        """
        List all sectors in a universe.

        Args:
            universe: Universe name

        Returns:
            List of sector names
        """
        sectors = self.get_sectors(universe)
        return list(sectors.keys())

    def create_custom_universe(
        self,
        name: str,
        tickers: List[str],
        description: str = "",
        sectors: Optional[Dict[str, List[str]]] = None
    ):
        """
        Create a custom stock universe.

        Args:
            name: Universe name
            tickers: List of tickers
            description: Universe description
            sectors: Optional sector mappings
        """
        universe_data = {
            "name": name,
            "description": description,
            "last_updated": datetime.now().isoformat(),
            "tickers": tickers,
            "sectors": sectors or {}
        }

        # Save to memory (not persisted to disk unless explicitly saved)
        self.universes[name] = universe_data

    def save_universe(self, name: str):
        """
        Save a universe to disk.

        Args:
            name: Universe name to save
        """
        if name not in self.universes:
            raise ValueError(f"Universe '{name}' not found")

        file_path = self.data_dir / f"{name}.json"

        with open(file_path, 'w') as f:
            json.dump(self.universes[name], f, indent=2)

        print(f"Universe '{name}' saved to {file_path}")

    def filter_tickers(
        self,
        universe: str = "sp100",
        sectors: Optional[List[str]] = None,
        exclude: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Filter tickers by criteria.

        Args:
            universe: Universe name
            sectors: Only include these sectors (None = all)
            exclude: Set of tickers to exclude

        Returns:
            Filtered list of tickers
        """
        if sectors:
            # Get tickers from specified sectors
            tickers = []
            for sector in sectors:
                tickers.extend(self.get_tickers_by_sector(sector, universe))
            # Remove duplicates
            tickers = list(set(tickers))
        else:
            # Get all tickers
            tickers = self.get_tickers(universe)

        # Apply exclusions
        if exclude:
            tickers = [t for t in tickers if t not in exclude]

        return tickers

    def get_universe_info(self, name: str = "sp100") -> Dict:
        """
        Get detailed information about a universe.

        Args:
            name: Universe name

        Returns:
            Dictionary with universe statistics
        """
        universe = self.get_universe(name)
        sectors = universe.get("sectors", {})

        return {
            "name": universe.get("name", name),
            "description": universe.get("description", ""),
            "total_tickers": len(universe.get("tickers", [])),
            "sectors": len(sectors),
            "sector_breakdown": {
                sector: len(tickers) for sector, tickers in sectors.items()
            },
            "last_updated": universe.get("last_updated", "Unknown")
        }


# Convenience function for quick access
def get_default_universe() -> StockUniverse:
    """Get the default stock universe instance."""
    return StockUniverse()


if __name__ == "__main__":
    # Example usage
    universe = StockUniverse()

    print("Available universes:", universe.list_universes())
    print("\nS&P 100 Info:")
    info = universe.get_universe_info("sp100")
    print(f"  Total stocks: {info['total_tickers']}")
    print(f"  Sectors: {info['sectors']}")

    print("\nTechnology stocks:")
    tech_stocks = universe.get_tickers_by_sector("Technology")
    print(f"  Count: {len(tech_stocks)}")
    print(f"  Examples: {tech_stocks[:5]}")

    print("\nAll sectors:")
    for sector in universe.list_sectors():
        count = len(universe.get_tickers_by_sector(sector))
        print(f"  {sector}: {count} stocks")
