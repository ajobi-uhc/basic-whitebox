import torch
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from main import (
    model,
    tokenizer,
    num_layers,
)


def get_statement_activation(question: str, answer: str, layer: int):
    """
    Extract activation at the final token position of a Q&A statement.
    Reads from the residual stream (layer input = resid_pre).

    Args:
        question: The question text
        answer: The answer text
        layer: Layer index to extract activation from

    Returns:
        numpy array: Activation vector at final token position [hidden_dim]
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

    # Tokenize to list of token IDs (nnsight handles the tensor conversion)
    tokens = tokenizer.encode(full_prompt)

    # Extract activation at final token position from residual stream
    with torch.no_grad():
        with model.trace(tokens):
            # Use layer input (resid_pre) for consistency
            activation = model.model.layers[layer].input[0].save()

    # Get final token position
    h = activation[-1, :].float().cpu().numpy()

    return h


def extract_truthful_steering_vectors(
    json_path: str,
    layers: list,
    output_path: str,
    max_samples: int = None
):
    """
    Extract steering vectors for truthfulness detection using difference-in-means.

    For each layer:
    1. Collect all truthful statement activations
    2. Collect all false statement activations
    3. Compute: mean(truthful) - mean(false)
    4. Normalize to unit vector

    Args:
        json_path: Path to truthful.json
        layers: List of layer indices to extract activations from
        output_path: Path to save the steering vectors (.npz)
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        Dictionary mapping layer indices to steering vectors
    """
    # Load truthful data
    with open(json_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"\nProcessing {len(data)} questions...")
    print("="*80)

    # Storage for all activations per layer (difference-in-means approach)
    layer_activations = {layer: {'true': [], 'false': []} for layer in layers}

    # Process each question
    for idx, item in enumerate(data):
        if (idx + 1) % 10 == 0:
            print(f"Processing question {idx + 1}/{len(data)}...")

        question = item['question']
        truthful_answer = item['positive']  # Truthful answer
        false_answer = item['negative']    # False answer

        # Collect activations for both true and false statements
        for layer in layers:
            true_activation = get_statement_activation(question, truthful_answer, layer)
            false_activation = get_statement_activation(question, false_answer, layer)

            layer_activations[layer]['true'].append(true_activation)
            layer_activations[layer]['false'].append(false_activation)

        # Clear memory periodically
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Compute difference-in-means steering vectors
    steering_vectors = {}
    print("\n" + "="*80)
    print("TRUTHFULNESS STEERING VECTORS (difference-in-means):")
    for layer in layers:
        if layer_activations[layer]['true'] and layer_activations[layer]['false']:
            # Stack into arrays [n_samples, hidden_dim]
            H_true = np.stack(layer_activations[layer]['true'])
            H_false = np.stack(layer_activations[layer]['false'])

            # Compute class means
            mu_true = H_true.mean(axis=0)
            mu_false = H_false.mean(axis=0)

            # Difference-in-means
            w = mu_true - mu_false

            # Normalize to unit vector
            w = w / (np.linalg.norm(w) + 1e-12)

            steering_vectors[layer] = w
            print(f"Layer {layer}: shape={w.shape}, norm={np.linalg.norm(w):.4f}, n_true={len(H_true)}, n_false={len(H_false)}")
        else:
            print(f"Layer {layer}: No samples collected")

    # Save to .npz file
    np.savez(output_path, **{f"layer_{layer}": steering_vectors[layer] for layer in layers if layer in steering_vectors})
    print(f"\nSaved steering vectors to {output_path}")
    print("="*80)

    return steering_vectors


def get_probe_score(
    question: str,
    answer: str,
    steering_vectors: dict,
    layer: int
):
    """
    Calculate probe score for a given question + answer using dot product.

    Args:
        question: The question/prompt
        answer: The answer text
        steering_vectors: Dictionary of steering vectors indexed by layer
        layer: The specific layer to probe

    Returns:
        float: Probe score (higher = more truthful)
    """
    if layer not in steering_vectors:
        raise ValueError(f"No steering vector available for layer {layer}")

    # Get activation at final token for this statement
    h = get_statement_activation(question, answer, layer)

    # Get steering vector (already normalized)
    w = steering_vectors[layer]

    # Compute dot product (projection onto steering vector)
    score = np.dot(h, w)

    return score


def evaluate_probe(
    json_path: str,
    steering_vectors: dict,
    layers: list,
    max_samples: int = None
):
    """
    Evaluate the steering vector probes on the truthful dataset.

    Args:
        json_path: Path to truthful.json
        steering_vectors: Dictionary of steering vectors indexed by layer
        layers: List of layers to evaluate
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation results for each layer
    """
    # Load truthful data
    with open(json_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"\nEvaluating on {len(data)} questions...")
    print("="*80)

    results = {layer: {
        'correct': 0,
        'total': 0,
        'scores_diff': [],
        'all_scores': [],  # All scores for AUC calculation
        'all_labels': []   # All labels (1=truthful, 0=false)
    } for layer in layers}

    for idx, item in enumerate(data):
        if (idx + 1) % 10 == 0:
            print(f"Evaluating question {idx + 1}/{len(data)}...")

        question = item['question']
        truthful_answer = item['positive']
        false_answer = item['negative']

        for layer in layers:
            # Get scores for both answers
            truthful_score = get_probe_score(question, truthful_answer, steering_vectors, layer)
            false_score = get_probe_score(question, false_answer, steering_vectors, layer)

            # Store all scores and labels for AUC
            results[layer]['all_scores'].append(truthful_score)
            results[layer]['all_labels'].append(1)  # Truthful
            results[layer]['all_scores'].append(false_score)
            results[layer]['all_labels'].append(0)  # False

            # Check if probe correctly identifies truthful answer (higher score)
            if truthful_score > false_score:
                results[layer]['correct'] += 1

            results[layer]['total'] += 1
            results[layer]['scores_diff'].append(truthful_score - false_score)

        # Clear memory periodically
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Calculate accuracies and AUC
    print("\n" + "="*80)
    print("EVALUATION RESULTS:")
    print("="*80)
    print(f"\n{'Layer':<10} {'Accuracy':<15} {'AUC':<15} {'Avg Score Diff':<20}")
    print("-" * 60)

    for layer in sorted(layers):
        if results[layer]['total'] == 0:
            print(f"{layer:<10} No data")
            continue

        accuracy = results[layer]['correct'] / results[layer]['total']
        avg_diff = np.mean(results[layer]['scores_diff']) if results[layer]['scores_diff'] else 0

        # Calculate AUC
        if len(results[layer]['all_scores']) > 0 and len(set(results[layer]['all_labels'])) > 1:
            auc = roc_auc_score(results[layer]['all_labels'], results[layer]['all_scores'])
        else:
            auc = 0.0

        results[layer]['accuracy'] = accuracy
        results[layer]['avg_score_diff'] = avg_diff
        results[layer]['auc'] = auc

        # Calculate score statistics
        truthful_scores = [s for s, l in zip(results[layer]['all_scores'], results[layer]['all_labels']) if l == 1]
        false_scores = [s for s, l in zip(results[layer]['all_scores'], results[layer]['all_labels']) if l == 0]

        results[layer]['truthful_mean'] = np.mean(truthful_scores) if truthful_scores else 0
        results[layer]['false_mean'] = np.mean(false_scores) if false_scores else 0
        results[layer]['truthful_std'] = np.std(truthful_scores) if truthful_scores else 0
        results[layer]['false_std'] = np.std(false_scores) if false_scores else 0

        print(f"{layer:<10} {accuracy:<15.4f} {auc:<15.4f} {avg_diff:<20.6f}")

    # Find best layer by AUC
    valid_layers = [l for l in results.keys() if results[l]['total'] > 0]
    if valid_layers:
        best_layer = max(valid_layers, key=lambda l: results[l]['auc'])
        print(f"\nBest performing layer (by AUC): {best_layer} (AUC: {results[best_layer]['auc']:.4f}, Accuracy: {results[best_layer]['accuracy']:.4f})")

        # Print detailed stats for best layer
        print(f"\nDetailed stats for layer {best_layer}:")
        print(f"  Truthful answers: mean={results[best_layer]['truthful_mean']:.4f}, std={results[best_layer]['truthful_std']:.4f}")
        print(f"  False answers:    mean={results[best_layer]['false_mean']:.4f}, std={results[best_layer]['false_std']:.4f}")

    return results


if __name__ == "__main__":
    # Configuration
    LAYERS_TO_PROBE = [i for i in range(0, num_layers, 4)]  # Every 4th layer
    MAX_SAMPLES_TRAIN = 150  # Number of samples to use for creating steering vectors
    MAX_SAMPLES_EVAL = 60   # Number of samples to use for evaluation (separate from training)
    OUTPUT_PATH = "truthfulness_steering_vectors.npz"

    print(f"\nProbing layers: {LAYERS_TO_PROBE}")
    print(f"Training samples: {MAX_SAMPLES_TRAIN}")
    print(f"Evaluation samples: {MAX_SAMPLES_EVAL}")
    print("Method: difference-in-means at final token position")

    # Extract steering vectors from first MAX_SAMPLES_TRAIN questions
    steering_vectors = extract_truthful_steering_vectors(
        json_path="truthful.json",
        layers=LAYERS_TO_PROBE,
        output_path=OUTPUT_PATH,
        max_samples=MAX_SAMPLES_TRAIN
    )

    # Evaluate on a separate set of questions (skip the training set)
    print(f"\n\nEvaluating on questions {MAX_SAMPLES_TRAIN + 1} to {MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL}...")

    # Create a temporary file with eval data
    with open("truthful.json", 'r') as f:
        all_data = json.load(f)

    eval_data = all_data[MAX_SAMPLES_TRAIN:MAX_SAMPLES_TRAIN + MAX_SAMPLES_EVAL]

    with open("truthful_eval.json", 'w') as f:
        json.dump(eval_data, f)

    # Evaluate
    results = evaluate_probe(
        json_path="truthful_eval.json",
        steering_vectors=steering_vectors,
        layers=LAYERS_TO_PROBE,
        max_samples=None  # Use all eval data
    )

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
