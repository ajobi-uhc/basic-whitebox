"""
Unified pipeline to extract and validate steering vectors.

This script runs:
1. Training: Extract steering vectors from training data
2. Validation: Test steering effectiveness on validation data
3. Report: Find and report the best performing layer
"""

import subprocess
import sys
import argparse
import json
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run complete steering vector extraction and validation pipeline"
    )

    # Data paths
    parser.add_argument("--train_data", type=str, default="steering_train_data.json",
                       help="Path to training data JSON")
    parser.add_argument("--val_data", type=str, default="steering_val_data.json",
                       help="Path to validation data JSON")

    # Training parameters
    parser.add_argument("--sample_size", type=int, default=128,
                       help="Number of training samples per category")
    parser.add_argument("--layer_step", type=int, default=2,
                       help="Extract from every Nth layer")

    # Validation parameters
    parser.add_argument("--strength", type=float, default=-1.0,
                       help="Steering strength for validation (negative reduces refusal)")
    parser.add_argument("--max_val_samples", type=int, default=10,
                       help="Max validation samples per category (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed validation results")

    # Output paths
    parser.add_argument("--output_vectors", type=str, default="steering_vectors.npz",
                       help="Output path for steering vectors")
    parser.add_argument("--output_results", type=str, default="validation_results.json",
                       help="Output path for validation results")
    parser.add_argument("--output_responses", type=str, default="validation_responses.txt",
                       help="Output file for all validation responses")

    # Pipeline control
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training (use existing vectors)")
    parser.add_argument("--skip_validation", action="store_true",
                       help="Skip validation")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STEERING VECTOR PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Training data: {args.train_data}")
    print(f"  Validation data: {args.val_data}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Layer step: {args.layer_step}")
    print(f"  Steering strength: {args.strength}")
    print(f"  Output vectors: {args.output_vectors}")
    print(f"  Output results: {args.output_results}")

    # Check if data files exist
    if not Path(args.train_data).exists():
        print(f"\n❌ Error: Training data not found at {args.train_data}")
        print("Please run download_steering_data.py first to create the datasets.")
        sys.exit(1)

    if not Path(args.val_data).exists():
        print(f"\n❌ Error: Validation data not found at {args.val_data}")
        print("Please run download_steering_data.py first to create the datasets.")
        sys.exit(1)

    # Step 1: Training (extract steering vectors)
    if not args.skip_training:
        train_cmd = [
            "python", "main.py",
            "--train_data", args.train_data,
            "--output", args.output_vectors,
            "--sample_size", str(args.sample_size),
            "--layer_step", str(args.layer_step)
        ]
        run_command(train_cmd, "STEP 1: Extracting Steering Vectors")
    else:
        print(f"\n⏭️  Skipping training (using existing vectors at {args.output_vectors})")

        if not Path(args.output_vectors).exists():
            print(f"\n❌ Error: Steering vectors not found at {args.output_vectors}")
            print("Cannot skip training without existing vectors.")
            sys.exit(1)

    # Step 2: Validation
    if not args.skip_validation:
        val_cmd = [
            "python", "validate.py",
            "--val_data", args.val_data,
            "--steering_vectors", args.output_vectors,
            "--strength", str(args.strength),
            "--max_samples", str(args.max_val_samples),
            "--output", args.output_results,
            "--responses_file", args.output_responses
        ]

        if args.verbose:
            val_cmd.append("--verbose")

        run_command(val_cmd, "STEP 2: Validating Steering Vectors")

        # Load and display final results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)

        with open(args.output_results, 'r') as f:
            results = json.load(f)

        best = results.get('best_layer')
        if best:
            print(f"\n🏆 Best Steered Layer: {best['layer']}")
            print(f"   Harmful Compliance: {best['harmful_compliance_rate']:.1%}")
            print(f"   Harmless Compliance: {best['harmless_compliance_rate']:.1%}")
            print(f"   Score: {best['score']:.1%}")

        print(f"\n📊 All Results:")
        print(f"{'Layer':<12} {'Harmful':<15} {'Harmless':<15} {'Score':<10}")
        print("-" * 60)
        for result in results['results']:
            layer_display = str(result['layer'])
            print(f"{layer_display:<12} "
                  f"{result['harmful_compliance_rate']:<15.1%} "
                  f"{result['harmless_compliance_rate']:<15.1%} "
                  f"{result['score']:<10.1%}")

        print(f"\n📄 Files created:")
        print(f"   - Results: {args.output_results}")
        print(f"   - Responses: {args.output_responses}")
    else:
        print(f"\n⏭️  Skipping validation")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
