#!/usr/bin/env python3
"""
Test script dla WronAI data collection
"""

import sys
import logging
from pathlib import Path

# Dodaj scripts do path
sys.path.append(str(Path(__file__).parent))

from collect_wronai_data_fixed import WronAICollector

def test_small_collection():
    """Test z małym rozmiarem danych."""
    print("🧪 Test zbierania danych...")

    collector = WronAICollector(
        output_dir="./test_data",
        target_size_gb=0.1  # 100MB dla testu
    )

    collector.run()

    # Sprawdź wyniki
    data_dir = Path("./test_data")
    files = list(data_dir.glob("*.jsonl"))

    print(f"✅ Utworzono {len(files)} plików:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f}MB")

if __name__ == "__main__":
    test_small_collection()
