#!/usr/bin/env python3
"""
Task Vector Extractor for Mergekit

This script extracts and saves task vectors by comparing fine-tuned models with their base model.
Task Vector = Fine-tuned Model - Base Model

Usage:
    python extract_task_vectors.py --base_model <base_model_path> --finetuned_model <finetuned_model_path> --output_dir <output_directory>
"""

import argparse
import os
import logging
from typing import Dict, Optional
import torch
import safetensors.torch
from tqdm import tqdm

from mergekit.common import ModelReference, ModelPath
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_task_vector(
    base_model_path: str,
    finetuned_model_path: str,
    output_dir: str,
    device: str = "cpu",
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None
):
    """
    Extract task vector by comparing fine-tuned model with base model.
    
    Args:
        base_model_path: Path to the base model
        finetuned_model_path: Path to the fine-tuned model
        output_dir: Directory to save the task vector
        device: Device to perform computation on
        trust_remote_code: Whether to trust remote code
        cache_dir: Cache directory for model downloads
    """
    
    # Create model references
    base_model_ref = ModelReference(model=ModelPath(path=base_model_path))
    finetuned_model_ref = ModelReference(model=ModelPath(path=finetuned_model_path))
    
    # Set up loader cache
    loader_cache = LoaderCache()
    options = MergeOptions(
        transformers_cache=cache_dir,
        trust_remote_code=trust_remote_code,
        device=device,
        cuda=(device != "cpu")
    )
    loader_cache.setup(options)
    
    # Get loaders for both models
    logging.info(f"Loading base model: {base_model_path}")
    base_loader = loader_cache.get(base_model_ref)
    
    logging.info(f"Loading fine-tuned model: {finetuned_model_path}")
    finetuned_loader = loader_cache.get(finetuned_model_ref)
    
    # Get all tensor keys from base model
    base_tensor_keys = set(base_loader.index.tensor_paths.keys())
    finetuned_tensor_keys = set(finetuned_loader.index.tensor_paths.keys())
    
    # Find common keys
    common_keys = base_tensor_keys.intersection(finetuned_tensor_keys)
    if not common_keys:
        raise RuntimeError("No common tensor keys found between models")
    
    logging.info(f"Found {len(common_keys)} common tensor keys")
    
    # Extract task vectors
    task_vectors = {}
    skipped_keys = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each tensor
    for key in tqdm(common_keys, desc="Extracting task vectors"):
        try:
            # Load tensors from both models
            base_tensor = base_loader.get_tensor(key, device=device)
            finetuned_tensor = finetuned_loader.get_tensor(key, device=device)
            
            # Check shape compatibility
            if base_tensor.shape != finetuned_tensor.shape:
                logging.warning(f"Shape mismatch for {key}: base {base_tensor.shape} vs finetuned {finetuned_tensor.shape}")
                if "embed" in key.lower():
                    # Handle embedding size mismatches
                    min_rows = min(base_tensor.shape[0], finetuned_tensor.shape[0])
                    min_cols = min(base_tensor.shape[1], finetuned_tensor.shape[1])
                    base_tensor = base_tensor[:min_rows, :min_cols]
                    finetuned_tensor = finetuned_tensor[:min_rows, :min_cols]
                    logging.info(f"Truncated {key} to {base_tensor.shape}")
                else:
                    skipped_keys.append(key)
                    continue
            
            # Compute task vector: delta = finetuned - base
            task_vector = finetuned_tensor - base_tensor
            
            # Store task vector
            task_vectors[key] = task_vector.cpu()
            
        except Exception as e:
            logging.error(f"Error processing {key}: {e}")
            skipped_keys.append(key)
    
    if skipped_keys:
        logging.warning(f"Skipped {len(skipped_keys)} tensors due to errors: {skipped_keys[:10]}...")
    
    # Save task vectors
    task_vector_path = os.path.join(output_dir, "task_vector.safetensors")
    logging.info(f"Saving task vectors to {task_vector_path}")
    safetensors.torch.save_file(task_vectors, task_vector_path)
    
    # Save metadata
    metadata = {
        "base_model": base_model_path,
        "finetuned_model": finetuned_model_path,
        "num_tensors": len(task_vectors),
        "skipped_tensors": len(skipped_keys),
        "device_used": device,
    }
    
    metadata_path = os.path.join(output_dir, "task_vector_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    # Calculate and save statistics
    stats = calculate_task_vector_stats(task_vectors)
    stats_path = os.path.join(output_dir, "task_vector_stats.txt")
    with open(stats_path, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logging.info(f"Task vector extraction completed. Saved {len(task_vectors)} tensors to {output_dir}")
    return task_vectors, metadata, stats


def calculate_task_vector_stats(task_vectors: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate statistics about the task vectors."""
    stats = {}
    
    total_params = sum(tv.numel() for tv in task_vectors.values())
    stats["total_parameters"] = total_params
    
    # Calculate magnitude statistics
    all_magnitudes = torch.cat([tv.abs().flatten() for tv in task_vectors.values()])
    stats["mean_abs_magnitude"] = all_magnitudes.mean().item()
    stats["std_abs_magnitude"] = all_magnitudes.std().item()
    stats["max_abs_magnitude"] = all_magnitudes.max().item()
    stats["min_abs_magnitude"] = all_magnitudes.min().item()
    
    # Calculate sparsity (percentage of near-zero values)
    threshold = 1e-6
    near_zero = (all_magnitudes < threshold).sum().item()
    stats["sparsity_ratio"] = near_zero / len(all_magnitudes)
    
    # Per-layer statistics
    layer_stats = {}
    for name, tensor in task_vectors.items():
        layer_stats[name] = {
            "shape": list(tensor.shape),
            "mean_abs": tensor.abs().mean().item(),
            "max_abs": tensor.abs().max().item(),
            "std": tensor.std().item(),
        }
    
    return stats


def load_task_vector(task_vector_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a previously saved task vector."""
    logging.info(f"Loading task vector from {task_vector_path}")
    task_vectors = safetensors.torch.load_file(task_vector_path, device=device)
    logging.info(f"Loaded task vector with {len(task_vectors)} tensors")
    return task_vectors


def apply_task_vector(
    base_model_path: str,
    task_vector_path: str,
    output_path: str,
    scaling_factor: float = 1.0,
    device: str = "cpu"
):
    """
    Apply a task vector to a base model to reconstruct the fine-tuned model.
    
    Args:
        base_model_path: Path to the base model
        task_vector_path: Path to the saved task vector
        output_path: Path to save the reconstructed model
        scaling_factor: Factor to scale the task vector (1.0 = full application)
        device: Device to perform computation on
    """
    
    # Load task vector
    task_vectors = load_task_vector(task_vector_path, device)
    
    # Load base model
    base_model_ref = ModelReference(model=ModelPath(path=base_model_path))
    loader_cache = LoaderCache()
    options = MergeOptions(device=device, cuda=(device != "cpu"))
    loader_cache.setup(options)
    base_loader = loader_cache.get(base_model_ref)
    
    # Apply task vectors
    reconstructed_tensors = {}
    
    for key in tqdm(task_vectors.keys(), desc="Applying task vectors"):
        try:
            base_tensor = base_loader.get_tensor(key, device=device)
            task_vector = task_vectors[key].to(device)
            
            # Apply: reconstructed = base + scaling_factor * task_vector
            reconstructed_tensor = base_tensor + scaling_factor * task_vector
            reconstructed_tensors[key] = reconstructed_tensor.cpu()
            
        except Exception as e:
            logging.error(f"Error applying task vector for {key}: {e}")
    
    # Save reconstructed model
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.safetensors")
    safetensors.torch.save_file(reconstructed_tensors, model_path)
    
    logging.info(f"Reconstructed model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and manipulate task vectors")
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--finetuned_model", help="Path to fine-tuned model (for extraction)")
    parser.add_argument("--task_vector", help="Path to saved task vector (for application)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Scaling factor for task vector application")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--cache_dir", help="Cache directory for model downloads")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--mode", choices=["extract", "apply"], default="extract", help="Mode: extract or apply task vectors")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.mode == "extract":
        if not args.finetuned_model:
            parser.error("--finetuned_model is required for extraction mode")
        
        extract_task_vector(
            base_model_path=args.base_model,
            finetuned_model_path=args.finetuned_model,
            output_dir=args.output_dir,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir
        )
    
    elif args.mode == "apply":
        if not args.task_vector:
            parser.error("--task_vector is required for application mode")
        
        apply_task_vector(
            base_model_path=args.base_model,
            task_vector_path=args.task_vector,
            output_path=args.output_dir,
            scaling_factor=args.scaling_factor,
            device=args.device
        )


if __name__ == "__main__":
    main()
