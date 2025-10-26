import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union

dotenv.load_dotenv()


def extract_contrastive_activations(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    json_path: Union[str, Path],
    layers: Union[int, List[int]],
    output_path: Union[str, Path],
    assistant_response: str = "Sure, I can help with that.",
) -> Dict[int, np.ndarray]:
    """
    Extract contrastive activation differences from prompt pairs.

    Args:
        model: NNsight LanguageModel
        tokenizer: HuggingFace tokenizer
        json_path: Path to JSON file with contrastive prompts
        layers: Layer index or list of layer indices to extract activations from
        output_path: Path to save the activation differences (.npz file)
        assistant_response: Default assistant response to prefill for activation extraction

    Returns:
        Dictionary mapping layer indices to mean difference vectors

    Expected JSON format:
        [
            {
                "positive": "prompt that elicits desired behavior",
                "negative": "prompt that elicits undesired behavior"
            },
            ...
        ]
    """
    # Load contrastive prompts
    with open(json_path, 'r') as f:
        contrastive_pairs = json.load(f)

    print(f"Loaded {len(contrastive_pairs)} contrastive prompt pairs")

    # Normalize layers to list
    if isinstance(layers, int):
        layers = [layers]

    # Storage for activation differences per layer
    layer_diffs = {layer: [] for layer in layers}

    # Process each contrastive pair
    for idx, pair in enumerate(contrastive_pairs):
        print(f"Processing pair {idx + 1}/{len(contrastive_pairs)}...")

        positive_prompt = pair["positive"]
        negative_prompt = pair["negative"]

        # Get activations for positive example
        pos_activations = _get_response_activations(
            model, tokenizer, positive_prompt, assistant_response, layers
        )

        # Get activations for negative example
        neg_activations = _get_response_activations(
            model, tokenizer, negative_prompt, assistant_response, layers
        )

        # Compute difference for each layer
        for layer in layers:
            diff = pos_activations[layer] - neg_activations[layer]
            layer_diffs[layer].append(diff)

    # Compute mean differences across all pairs for each layer
    mean_diffs = {}
    for layer in layers:
        mean_diffs[layer] = np.mean(layer_diffs[layer], axis=0)
        print(f"Layer {layer}: Mean diff shape = {mean_diffs[layer].shape}")

    # Save to .npz file
    np.savez(output_path, **{f"layer_{layer}": mean_diffs[layer] for layer in layers})
    print(f"Saved activation differences to {output_path}")

    return mean_diffs


def _get_response_activations(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    assistant_response: str,
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Get activations on assistant response tokens by prefilling.

    Args:
        model: NNsight LanguageModel
        tokenizer: HuggingFace tokenizer
        user_prompt: User message prompt
        assistant_response: Assistant response to prefill
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to mean activations across response tokens
    """
    # Format full conversation with assistant response
    chat = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    # Tokenize the full conversation
    tokens = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)

    # Find where assistant tokens start
    user_chat = [{"role": "user", "content": user_prompt}]
    user_tokens = tokenizer.apply_chat_template(user_chat, tokenize=True, add_generation_prompt=True)
    assistant_start_idx = len(user_tokens)

    # Run forward pass and collect activations
    layer_activations = {}

    with model.trace(tokens):
        for layer in layers:
            # Get residual stream activations at this layer
            layer_activations[layer] = model.model.layers[layer].output[0].save()

    # Extract and average over assistant tokens only
    mean_activations = {}
    for layer in layers:
        # Get activations for assistant tokens
        assistant_activations = layer_activations[layer][0, assistant_start_idx:, :]
        # Average over token dimension
        mean_activations[layer] = assistant_activations.mean(dim=0).cpu().numpy()

    return mean_activations


def load_steering_vectors(npz_path: Union[str, Path]) -> Dict[int, np.ndarray]:
    """
    Load steering vectors from saved .npz file.

    Args:
        npz_path: Path to .npz file with steering vectors

    Returns:
        Dictionary mapping layer indices to steering vectors
    """
    data = np.load(npz_path)
    steering_vectors = {}

    for key in data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            steering_vectors[layer_idx] = data[key]

    print(f"Loaded steering vectors for layers: {sorted(steering_vectors.keys())}")
    return steering_vectors