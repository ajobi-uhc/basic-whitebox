"""
Download and prepare training and validation data for steering vector extraction.

Creates two JSON files:
- steering_train_data.json: 128 harmful + 128 harmless instructions
- steering_val_data.json: 32 harmful + 32 harmless instructions
"""

import json
import random
import requests
import csv
from pathlib import Path
from typing import List, Dict
import io

# Set random seed for reproducibility
random.seed(42)


def download_advbench() -> List[str]:
    """Download harmful behaviors from ADVBENCH."""
    print("Downloading ADVBENCH...")
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    response.raise_for_status()

    # Parse CSV
    csv_file = io.StringIO(response.text)
    reader = csv.DictReader(csv_file)
    behaviors = [row['goal'] for row in reader]
    print(f"  Downloaded {len(behaviors)} harmful behaviors from ADVBENCH")
    return behaviors


def download_harmbench_val() -> List[str]:
    """Download validation set from HARMBENCH (standard behaviors only)."""
    print("Downloading HARMBENCH validation set...")
    # HarmBench validation set
    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_val.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()

        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)

        # Filter for standard behaviors only (exclude context-requiring or copyright)
        behaviors = []
        for row in reader:
            # Check if it's a standard behavior (not context-based or copyright)
            if row.get('SemanticCategory', '').lower() not in ['copyright', 'context']:
                behaviors.append(row['Behavior'])

        print(f"  Downloaded {len(behaviors)} standard harmful behaviors from HARMBENCH")
        return behaviors
    except Exception as e:
        print(f"  Warning: Could not download HARMBENCH validation set: {e}")
        print("  Will use ADVBENCH for validation instead")
        return []


def download_alpaca() -> List[str]:
    """Download harmless instructions from ALPACA."""
    print("Downloading ALPACA...")
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    # Extract instructions (some may have additional input context)
    instructions = []
    for item in data:
        instruction = item['instruction']
        if item.get('input', '').strip():
            instruction = f"{instruction}\n{item['input']}"
        instructions.append(instruction)

    print(f"  Downloaded {len(instructions)} harmless instructions from ALPACA")
    return instructions


def create_training_data(harmful_samples: List[str], harmless_samples: List[str]) -> Dict:
    """Create training data JSON structure."""
    return {
        "harmful": harmful_samples[:128],
        "harmless": harmless_samples[:128]
    }


def create_validation_data(harmful_samples: List[str], harmless_samples: List[str]) -> Dict:
    """Create validation data JSON structure."""
    return {
        "harmful": harmful_samples[:32],
        "harmless": harmless_samples[:32]
    }


def main():
    print("=" * 60)
    print("Downloading Steering Vector Training & Validation Data")
    print("=" * 60)
    print()

    # Download datasets
    print("Step 1: Downloading harmful instruction datasets...")
    print("-" * 60)
    advbench_harmful = download_advbench()
    harmbench_val_harmful = download_harmbench_val()

    print()
    print("Step 2: Downloading harmless instruction datasets...")
    print("-" * 60)
    alpaca_harmless = download_alpaca()

    # Shuffle datasets
    print()
    print("Step 3: Shuffling datasets...")
    print("-" * 60)
    random.shuffle(advbench_harmful)
    random.shuffle(alpaca_harmless)
    if harmbench_val_harmful:
        random.shuffle(harmbench_val_harmful)

    # Create training data (128 harmful + 128 harmless)
    print()
    print("Step 4: Creating training dataset...")
    print("-" * 60)
    train_data = create_training_data(advbench_harmful, alpaca_harmless)
    print(f"  Training set: {len(train_data['harmful'])} harmful + {len(train_data['harmless'])} harmless")

    # Create validation data (32 harmful + 32 harmless)
    print()
    print("Step 5: Creating validation dataset...")
    print("-" * 60)

    # Use HarmBench for validation if available, otherwise use remaining ADVBENCH
    if harmbench_val_harmful and len(harmbench_val_harmful) >= 32:
        val_harmful = harmbench_val_harmful
        print("  Using HARMBENCH for validation harmful set")
    else:
        # Use ADVBENCH samples not in training set
        val_harmful = advbench_harmful[128:]
        print("  Using ADVBENCH (samples 129+) for validation harmful set")

    # Use ALPACA samples not in training set for validation
    val_harmless = alpaca_harmless[128:]

    val_data = create_validation_data(val_harmful, val_harmless)
    print(f"  Validation set: {len(val_data['harmful'])} harmful + {len(val_data['harmless'])} harmless")

    # Save to JSON files
    print()
    print("Step 6: Saving to JSON files...")
    print("-" * 60)

    train_file = "steering_train_data.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved training data to: {train_file}")

    val_file = "steering_val_data.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved validation data to: {val_file}")

    # Print sample examples
    print()
    print("=" * 60)
    print("Sample Examples")
    print("=" * 60)
    print()
    print("TRAINING - Harmful examples (first 3):")
    for i, example in enumerate(train_data['harmful'][:3], 1):
        print(f"  {i}. {example}")

    print()
    print("TRAINING - Harmless examples (first 3):")
    for i, example in enumerate(train_data['harmless'][:3], 1):
        print(f"  {i}. {example}")

    print()
    print("VALIDATION - Harmful examples (first 3):")
    for i, example in enumerate(val_data['harmful'][:3], 1):
        print(f"  {i}. {example}")

    print()
    print("VALIDATION - Harmless examples (first 3):")
    for i, example in enumerate(val_data['harmless'][:3], 1):
        print(f"  {i}. {example}")

    print()
    print("=" * 60)
    print("✓ Done! Files created:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
