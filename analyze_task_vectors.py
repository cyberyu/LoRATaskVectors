#!/usr/bin/env python3
"""
Task Vector Analyzer

This script analyzes extracted task vectors to provide insights about model differences,
similarities, and potential merge strategies.
"""

import argparse
import os
import logging
import yaml
from typing import Dict, List, Tuple
import torch
import safetensors.torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd


def load_task_vectors(directory: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load all task vectors from a directory."""
    task_vectors = {}
    
    for filename in os.listdir(directory):
        if filename.startswith("task_vector_") and filename.endswith(".safetensors"):
            model_name = filename.replace("task_vector_", "").replace(".safetensors", "")
            filepath = os.path.join(directory, filename)
            
            logging.info(f"Loading task vector for {model_name}")
            task_vectors[model_name] = safetensors.torch.load_file(filepath)
    
    return task_vectors


def calculate_similarity_matrix(task_vectors: Dict[str, Dict[str, torch.Tensor]]) -> np.ndarray:
    """Calculate cosine similarity matrix between task vectors."""
    model_names = list(task_vectors.keys())
    n_models = len(model_names)
    similarity_matrix = np.zeros((n_models, n_models))
    
    # Flatten each model's task vectors
    flattened_vectors = {}
    for model_name, tv_dict in task_vectors.items():
        # Concatenate all tensors for this model
        flattened = torch.cat([tv.flatten() for tv in tv_dict.values()])
        flattened_vectors[model_name] = flattened.numpy()
    
    # Calculate pairwise similarities
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity
                vec_i = flattened_vectors[model_i]
                vec_j = flattened_vectors[model_j]
                similarity = 1 - cosine(vec_i, vec_j)
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix, model_names


def analyze_layer_differences(task_vectors: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
    """Analyze differences at the layer level."""
    analysis = {}
    
    # Get common layers across all models
    all_layers = set()
    for tv_dict in task_vectors.values():
        all_layers.update(tv_dict.keys())
    
    common_layers = all_layers
    for tv_dict in task_vectors.values():
        common_layers = common_layers.intersection(set(tv_dict.keys()))
    
    logging.info(f"Found {len(common_layers)} common layers")
    
    # Analyze each layer
    layer_analysis = {}
    for layer_name in common_layers:
        layer_stats = {}
        
        for model_name, tv_dict in task_vectors.items():
            if layer_name in tv_dict:
                tensor = tv_dict[layer_name]
                layer_stats[model_name] = {
                    'mean_abs': tensor.abs().mean().item(),
                    'std': tensor.std().item(),
                    'max_abs': tensor.abs().max().item(),
                    'l2_norm': tensor.norm().item(),
                    'sparsity': (tensor.abs() < 1e-6).float().mean().item()
                }
        
        layer_analysis[layer_name] = layer_stats
    
    analysis['layer_analysis'] = layer_analysis
    analysis['common_layers'] = list(common_layers)
    
    return analysis


def find_most_different_layers(task_vectors: Dict[str, Dict[str, torch.Tensor]], top_k: int = 10) -> List[Tuple[str, float]]:
    """Find layers with the highest variance across models."""
    layer_variances = []
    
    # Get common layers
    common_layers = set(task_vectors[list(task_vectors.keys())[0]].keys())
    for tv_dict in task_vectors.values():
        common_layers = common_layers.intersection(set(tv_dict.keys()))
    
    for layer_name in common_layers:
        # Calculate variance in L2 norms across models
        l2_norms = []
        for tv_dict in task_vectors.values():
            if layer_name in tv_dict:
                l2_norms.append(tv_dict[layer_name].norm().item())
        
        if len(l2_norms) > 1:
            variance = np.var(l2_norms)
            layer_variances.append((layer_name, variance))
    
    # Sort by variance (descending)
    layer_variances.sort(key=lambda x: x[1], reverse=True)
    
    return layer_variances[:top_k]


def create_visualizations(task_vectors: Dict[str, Dict[str, torch.Tensor]], output_dir: str):
    """Create visualization plots for task vector analysis."""
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Similarity matrix heatmap
    similarity_matrix, model_names = calculate_similarity_matrix(task_vectors)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=model_names, 
                yticklabels=model_names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.3f')
    plt.title('Task Vector Similarity Matrix (Cosine Similarity)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'similarity_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Magnitude comparison
    model_magnitudes = {}
    for model_name, tv_dict in task_vectors.items():
        total_magnitude = sum(tv.norm().item() for tv in tv_dict.values())
        model_magnitudes[model_name] = total_magnitude
    
    plt.figure(figsize=(10, 6))
    names = list(model_magnitudes.keys())
    values = list(model_magnitudes.values())
    bars = plt.bar(names, values, color=['skyblue', 'lightgreen', 'lightcoral'][:len(names)])
    plt.title('Total Task Vector Magnitude by Model')
    plt.ylabel('L2 Norm')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'magnitude_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Layer-wise analysis for most different layers
    most_different = find_most_different_layers(task_vectors, top_k=5)
    
    if most_different:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (layer_name, variance) in enumerate(most_different[:5]):
            ax = axes[idx]
            
            layer_norms = []
            model_names_for_layer = []
            
            for model_name, tv_dict in task_vectors.items():
                if layer_name in tv_dict:
                    layer_norms.append(tv_dict[layer_name].norm().item())
                    model_names_for_layer.append(model_name)
            
            bars = ax.bar(model_names_for_layer, layer_norms, 
                         color=['skyblue', 'lightgreen', 'lightcoral'][:len(model_names_for_layer)])
            ax.set_title(f'{layer_name[:30]}...\n(Variance: {variance:.2e})')
            ax.set_ylabel('L2 Norm')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, layer_norms):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                       f'{value:.1e}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplot
        if len(most_different) < 6:
            axes[5].set_visible(False)
        
        plt.suptitle('Most Different Layers Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'most_different_layers.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Visualizations saved to {plots_dir}")


def generate_merge_recommendations(task_vectors: Dict[str, Dict[str, torch.Tensor]], 
                                 similarity_matrix: np.ndarray, 
                                 model_names: List[str]) -> Dict:
    """Generate merge strategy recommendations based on task vector analysis."""
    
    recommendations = {}
    
    # Calculate average similarity
    n = len(model_names)
    avg_similarity = (similarity_matrix.sum() - n) / (n * (n - 1))  # Exclude diagonal
    
    recommendations['average_similarity'] = avg_similarity
    
    # Recommend merge methods based on similarity
    if avg_similarity > 0.8:
        recommendations['recommended_method'] = 'linear'
        recommendations['reason'] = 'High similarity between models - simple linear interpolation should work well'
        recommendations['suggested_weights'] = 'equal'
    elif avg_similarity > 0.5:
        recommendations['recommended_method'] = 'ties'
        recommendations['reason'] = 'Moderate similarity - TIES can handle conflicts while preserving individual strengths'
        recommendations['suggested_density'] = 0.7
    else:
        recommendations['recommended_method'] = 'breadcrumbs_ties'
        recommendations['reason'] = 'Low similarity - dual pruning approach to remove noise and outliers'
        recommendations['suggested_gamma'] = 0.1
        recommendations['suggested_density'] = 0.6
    
    # Find most and least similar pairs
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix + np.eye(n) * -2), similarity_matrix.shape)
    min_sim_idx = np.unravel_index(np.argmin(similarity_matrix + np.eye(n) * 2), similarity_matrix.shape)
    
    recommendations['most_similar_pair'] = (model_names[max_sim_idx[0]], model_names[max_sim_idx[1]])
    recommendations['least_similar_pair'] = (model_names[min_sim_idx[0]], model_names[min_sim_idx[1]])
    recommendations['max_similarity'] = similarity_matrix[max_sim_idx]
    recommendations['min_similarity'] = similarity_matrix[min_sim_idx]
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze extracted task vectors")
    parser.add_argument("--input_dir", required=True, help="Directory containing extracted task vectors")
    parser.add_argument("--output_dir", help="Output directory for analysis (defaults to input_dir/analysis)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load task vectors
    logging.info(f"Loading task vectors from {args.input_dir}")
    task_vectors = load_task_vectors(args.input_dir)
    
    if not task_vectors:
        logging.error("No task vectors found!")
        return
    
    logging.info(f"Loaded task vectors for {len(task_vectors)} models: {list(task_vectors.keys())}")
    
    # Calculate similarity matrix
    similarity_matrix, model_names = calculate_similarity_matrix(task_vectors)
    
    # Analyze layer differences
    layer_analysis = analyze_layer_differences(task_vectors)
    
    # Find most different layers
    most_different_layers = find_most_different_layers(task_vectors)
    
    # Generate recommendations
    recommendations = generate_merge_recommendations(task_vectors, similarity_matrix, model_names)
    
    # Save analysis results
    results = {
        'model_names': model_names,
        'similarity_matrix': similarity_matrix.tolist(),
        'layer_analysis': layer_analysis,
        'most_different_layers': [(layer, float(variance)) for layer, variance in most_different_layers],
        'recommendations': recommendations
    }
    
    results_path = os.path.join(args.output_dir, "task_vector_analysis_results.yml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, indent=2, default_flow_style=False)
    
    # Create summary report
    report_path = os.path.join(args.output_dir, "analysis_summary.txt")
    with open(report_path, "w") as f:
        f.write("Task Vector Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Models analyzed: {', '.join(model_names)}\n")
        f.write(f"Average similarity: {recommendations['average_similarity']:.4f}\n\n")
        
        f.write("Similarity Matrix:\n")
        for i, name_i in enumerate(model_names):
            f.write(f"{name_i:15}")
            for j, name_j in enumerate(model_names):
                f.write(f"{similarity_matrix[i,j]:8.4f}")
            f.write("\n")
        f.write("\n")
        
        f.write("Recommendations:\n")
        f.write(f"- Recommended merge method: {recommendations['recommended_method']}\n")
        f.write(f"- Reason: {recommendations['reason']}\n")
        
        if 'suggested_weights' in recommendations:
            f.write(f"- Suggested weights: {recommendations['suggested_weights']}\n")
        if 'suggested_density' in recommendations:
            f.write(f"- Suggested density: {recommendations['suggested_density']}\n")
        if 'suggested_gamma' in recommendations:
            f.write(f"- Suggested gamma: {recommendations['suggested_gamma']}\n")
        
        f.write(f"\nMost similar pair: {recommendations['most_similar_pair']} (similarity: {recommendations['max_similarity']:.4f})\n")
        f.write(f"Least similar pair: {recommendations['least_similar_pair']} (similarity: {recommendations['min_similarity']:.4f})\n")
        
        f.write(f"\nMost different layers (top 10):\n")
        for i, (layer, variance) in enumerate(most_different_layers[:10]):
            f.write(f"{i+1:2d}. {layer[:60]:60} (variance: {variance:.2e})\n")
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(task_vectors, args.output_dir)
        except ImportError as e:
            logging.warning(f"Could not create plots: {e}. Install matplotlib and seaborn to enable plotting.")
    
    logging.info(f"Analysis complete. Results saved to {args.output_dir}")
    logging.info(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
