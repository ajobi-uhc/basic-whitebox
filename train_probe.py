import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

dotenv.load_dotenv()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-32B"

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")
model = LanguageModel(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded!")

# Get model info
num_layers = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"Number of layers: {num_layers}")
print(f"Hidden size: {hidden_size}")


def extract_activations_at_position(tokens: list, position: int, layers: list):
    """
    Extract activations at a specific token position.

    Args:
        tokens: List of token IDs
        position: Token position to extract activations from
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to activations at the specified position
    """
    layer_activations = {}

    with torch.no_grad():
        with model.trace(tokens):
            for layer in layers:
                layer_activations[layer] = model.model.layers[layer].output[0].save()

    # Extract from the specified position
    activations = {}
    for layer in layers:
        activations[layer] = layer_activations[layer][position, :].float().cpu().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    return activations


def get_answer_activations(question: str, answer: str, layers: list):
    """
    Extract activations at the last token of the answer.

    Args:
        question: The question text
        answer: The answer choice (e.g., "(A)" or "(B)")
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to activations at the last answer token
    """
    # Build chat with question and answer
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

    # Apply chat template
    full_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    # Tokenize
    tokens = tokenizer.encode(full_prompt)

    # Find the <|im_end|> token to get the last answer token
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]

    # Find all positions where im_end_token appears and get the last one
    im_end_positions = [i for i, t in enumerate(tokens) if t == im_end_token]
    if not im_end_positions:
        raise ValueError(f"Could not find <|im_end|> token")

    # Extract at the token right before the last <|im_end|>
    extract_position = im_end_positions[-1] - 1
    print(f"Extracting at position {extract_position} (token before <|im_end|>)")

    return extract_activations_at_position(tokens, extract_position, layers)


def load_truthful_data(json_path: str):
    """Load the truthful.json dataset."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_probe_dataset(
    json_path: str,
    layers: list,
    max_samples: int = None
):
    """
    Extract activations for probe training from truthful.json.

    Args:
        json_path: Path to truthful.json
        layers: List of layer indices to extract activations from
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        Dictionary with activations and labels for each layer
    """
    data = load_truthful_data(json_path)

    if max_samples:
        data = data[:max_samples]

    print(f"\nProcessing {len(data)} questions...")
    print("="*80)

    # Storage for each layer: list of (activation, label) tuples
    layer_datasets = {layer: {'activations': [], 'labels': []} for layer in layers}

    for idx, item in enumerate(data):
        if (idx + 1) % 10 == 0:
            print(f"Processing question {idx + 1}/{len(data)}...")

        question = item['question']
        positive_answer = item['positive']  # Truthful answer
        negative_answer = item['negative']  # False answer

        # Extract activations for truthful answer (label = 1)
        try:
            truthful_activations = get_answer_activations(question, positive_answer, layers)
            for layer in layers:
                layer_datasets[layer]['activations'].append(truthful_activations[layer])
                layer_datasets[layer]['labels'].append(1)  # Truthful
        except Exception as e:
            print(f"Error processing truthful answer for question {idx}: {e}")
            continue

        # Extract activations for false answer (label = 0)
        try:
            false_activations = get_answer_activations(question, negative_answer, layers)
            for layer in layers:
                layer_datasets[layer]['activations'].append(false_activations[layer])
                layer_datasets[layer]['labels'].append(0)  # False
        except Exception as e:
            print(f"Error processing false answer for question {idx}: {e}")
            continue

        # Clear memory periodically
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Convert to numpy arrays
    for layer in layers:
        layer_datasets[layer]['activations'] = np.array(layer_datasets[layer]['activations'])
        layer_datasets[layer]['labels'] = np.array(layer_datasets[layer]['labels'])
        print(f"\nLayer {layer}: {len(layer_datasets[layer]['labels'])} samples")
        print(f"  Activations shape: {layer_datasets[layer]['activations'].shape}")
        print(f"  Truthful samples: {np.sum(layer_datasets[layer]['labels'])}")
        print(f"  False samples: {len(layer_datasets[layer]['labels']) - np.sum(layer_datasets[layer]['labels'])}")

    return layer_datasets


def train_probes(layer_datasets: dict, test_size: float = 0.2, random_state: int = 42):
    """
    Train logistic regression probes for each layer.

    Args:
        layer_datasets: Dictionary with activations and labels for each layer
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of trained probes and their accuracies
    """
    probes = {}
    results = {}

    print("\n" + "="*80)
    print("TRAINING PROBES")
    print("="*80)

    for layer in sorted(layer_datasets.keys()):
        print(f"\n--- Layer {layer} ---")

        X = layer_datasets[layer]['activations']
        y = layer_datasets[layer]['labels']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train probe
        probe = LogisticRegression(max_iter=1000, random_state=random_state)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, probe.predict(X_train))
        test_acc = accuracy_score(y_test, probe.predict(X_test))

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        # Detailed classification report
        y_pred = probe.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['False', 'Truthful']))

        probes[layer] = probe
        results[layer] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

    return probes, results


def save_probes(probes: dict, output_path: str):
    """Save trained probes to disk."""
    with open(output_path, 'wb') as f:
        pickle.dump(probes, f)
    print(f"\nSaved probes to {output_path}")


def load_probes(probe_path: str):
    """Load trained probes from disk."""
    with open(probe_path, 'rb') as f:
        probes = pickle.load(f)
    print(f"Loaded probes from {probe_path}")
    return probes


if __name__ == "__main__":
    # Configuration
    LAYERS_TO_PROBE = [i for i in range(0, num_layers, 4)]  # Every 4th layer
    MAX_SAMPLES = 200  # Limit samples for faster training (set to None for all)
    OUTPUT_PATH = "lie_detection_probes.pkl"

    print(f"\nProbing layers: {LAYERS_TO_PROBE}")
    print(f"Max samples per question: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")

    # Extract dataset
    layer_datasets = extract_probe_dataset(
        json_path="truthful.json",
        layers=LAYERS_TO_PROBE,
        max_samples=MAX_SAMPLES
    )

    # Save extracted activations for later use
    print("\nSaving extracted activations...")
    np.savez("probe_activations.npz", **{
        f"layer_{layer}_X": layer_datasets[layer]['activations']
        for layer in LAYERS_TO_PROBE
    }, **{
        f"layer_{layer}_y": layer_datasets[layer]['labels']
        for layer in LAYERS_TO_PROBE
    })
    print("Saved to probe_activations.npz")

    # Train probes
    probes, results = train_probes(layer_datasets)

    # Save probes
    save_probes(probes, OUTPUT_PATH)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Layer':<10} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 40)
    for layer in sorted(results.keys()):
        train_acc = results[layer]['train_acc']
        test_acc = results[layer]['test_acc']
        print(f"{layer:<10} {train_acc:<15.4f} {test_acc:<15.4f}")

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: results[l]['test_acc'])
    best_test_acc = results[best_layer]['test_acc']
    print(f"\nBest performing layer: {best_layer} (Test Acc: {best_test_acc:.4f})")
