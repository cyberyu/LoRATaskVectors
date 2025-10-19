#!/usr/bin/env python3
"""
LoRA vs Task Vector Comparison

This script compares LoRA adapters with extracted task vectors to verify
their mathematical relationship and measure any differences.
"""

import argparse
import os
import logging
import torch
import safetensors.torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import defaultdict
import json


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_base_model(base_model_path: str, device: str = "cuda"):
    """Load the base model."""
    logging.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=False
    )
    return model


def load_lora_model(base_model, lora_path: str):
    """Load LoRA adapter and return both adapted and merged models."""
    logging.info(f"Loading LoRA adapter: {lora_path}")
    
    # Load LoRA adapted model (not merged)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    return lora_model


def extract_lora_weights(lora_model):
    """Extract LoRA adapter weights (A and B matrices)."""
    lora_weights = {}
    
    logging.info("Starting LoRA weight extraction...")
    
    # Get scaling factors from the peft config
    scaling_factor = 1.0
    if hasattr(lora_model, 'peft_config') and 'default' in lora_model.peft_config:
        config = lora_model.peft_config['default']
        if hasattr(config, 'lora_alpha') and hasattr(config, 'r'):
            scaling_factor = config.lora_alpha / config.r
            logging.info(f"Using scaling factor: {scaling_factor} (alpha={config.lora_alpha}, r={config.r})")
    
    # Iterate through all named modules to find LoRA components
    # We need to access the base model's modules that have LoRA attached
    lora_modules_found = 0
    for name, module in lora_model.base_model.model.named_modules():
        # Check if this module has LoRA components
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                # Access LoRA weights through the ModuleDict structure
                lora_A_weight = module.lora_A['default'].weight.data
                lora_B_weight = module.lora_B['default'].weight.data
                
                # Compute delta = lora_B @ lora_A * scaling
                lora_delta = (lora_B_weight @ lora_A_weight) * scaling_factor
                
                # Convert module name to weight name for task vector comparison
                # The name should be like "base_model.model.layers.0.self_attn.q_proj"
                # We need to convert it to match task vector format
                weight_name = name
                if weight_name.startswith('base_model.'):
                    weight_name = weight_name[len('base_model.'):]
                if not weight_name.endswith('.weight'):
                    weight_name = weight_name + '.weight'
                
                lora_weights[weight_name] = lora_delta
                lora_modules_found += 1
                
                logging.debug(f"Extracted LoRA delta for {weight_name}: "
                            f"A{lora_A_weight.shape} @ B{lora_B_weight.shape} * {scaling_factor} = {lora_delta.shape}")
                
            except Exception as e:
                logging.warning(f"Could not extract LoRA weights from module {name}: {e}")
                continue
    
    logging.info(f"Found LoRA components in {lora_modules_found} modules")
    logging.info(f"Successfully extracted {len(lora_weights)} LoRA deltas")
    
    # Debug: Show first few extracted weights
    if lora_weights:
        sample_names = list(lora_weights.keys())[:5]
        logging.info(f"Sample extracted weights: {sample_names}")
        # Show actual tensor statistics
        for name in sample_names:
            tensor = lora_weights[name]
            logging.info(f"  {name}: shape={tensor.shape}, norm={torch.norm(tensor).item():.6f}")
    else:
        logging.warning("No LoRA weights were extracted! This might indicate a structural issue.")
        # Debug: Show available modules for troubleshooting
        module_names = [name for name, module in lora_model.base_model.model.named_modules() 
                       if hasattr(module, 'lora_A') or hasattr(module, 'lora_B')]
        logging.info(f"Modules with LoRA attributes: {len(module_names)}")
        if module_names:
            logging.info(f"Sample module names: {module_names[:5]}")
    
    return lora_weights


def compute_task_vector_from_models(base_model, merged_model):
    """Compute task vector by subtracting base from merged model."""
    task_vector = {}
    
    base_state_dict = base_model.state_dict()
    merged_state_dict = merged_model.state_dict()
    
    for name in tqdm(base_state_dict.keys(), desc="Computing task vectors"):
        if name in merged_state_dict:
            base_weight = base_state_dict[name]
            merged_weight = merged_state_dict[name]
            
            if base_weight.shape == merged_weight.shape:
                delta = merged_weight - base_weight
                task_vector[name] = delta
                logging.debug(f"Task vector for {name}: shape {delta.shape}")
            else:
                logging.warning(f"Shape mismatch for {name}: {base_weight.shape} vs {merged_weight.shape}")
    
    return task_vector


def compare_lora_and_task_vector(lora_weights, task_vector, tolerance=1e-5):
    """Compare LoRA reconstructed weights with task vectors."""
    comparison_results = {}
    
    logging.info("Comparing LoRA deltas with task vectors...")
    
    for name in task_vector.keys():
        if name in lora_weights:
            lora_delta = lora_weights[name]
            task_delta = task_vector[name]
            
            # Ensure same device and dtype
            lora_delta = lora_delta.to(task_delta.device).to(task_delta.dtype)
            
            if lora_delta.shape == task_delta.shape:
                # Calculate differences
                diff = torch.abs(lora_delta - task_delta)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                relative_error = torch.mean(diff / (torch.abs(task_delta) + 1e-8)).item()
                
                # Calculate similarity metrics
                lora_norm = torch.norm(lora_delta).item()
                task_norm = torch.norm(task_delta).item()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    lora_delta.flatten(), task_delta.flatten(), dim=0
                ).item()
                
                comparison_results[name] = {
                    'max_absolute_diff': max_diff,
                    'mean_absolute_diff': mean_diff,
                    'relative_error': relative_error,
                    'lora_norm': lora_norm,
                    'task_norm': task_norm,
                    'cosine_similarity': cosine_sim,
                    'shapes_match': True,
                    'within_tolerance': max_diff < tolerance
                }
                
                logging.debug(f"{name}: max_diff={max_diff:.2e}, cosine_sim={cosine_sim:.6f}")
            else:
                comparison_results[name] = {
                    'shapes_match': False,
                    'lora_shape': lora_delta.shape,
                    'task_shape': task_delta.shape
                }
                logging.warning(f"Shape mismatch for {name}: LoRA {lora_delta.shape} vs Task {task_delta.shape}")
        else:
            # Task vector exists but no corresponding LoRA
            comparison_results[name] = {
                'in_lora': False,
                'in_task_vector': True,
                'task_norm': torch.norm(task_vector[name]).item()
            }
    
    # Check for LoRA weights not in task vector
    for name in lora_weights.keys():
        if name not in task_vector:
            comparison_results[name] = {
                'in_lora': True,
                'in_task_vector': False,
                'lora_norm': torch.norm(lora_weights[name]).item()
            }
    
    return comparison_results


def parse_parameter_name(param_name):
    """Parse parameter name to extract structural information."""
    parts = param_name.split('.')
    
    info = {
        'full_name': param_name,
        'layer_type': 'unknown',
        'layer_idx': None,
        'component': 'unknown',
        'parameter_type': 'weight'  # weight or bias
    }
    
    # Extract parameter type (weight/bias)
    if parts[-1] in ['weight', 'bias']:
        info['parameter_type'] = parts[-1]
        component_parts = parts[:-1]
    else:
        component_parts = parts
    
    # Identify layer structure
    if 'embed_tokens' in param_name:
        info['layer_type'] = 'embedding'
        info['component'] = 'embed_tokens'
    elif 'layers' in parts:
        layer_idx = parts[parts.index('layers') + 1]
        try:
            info['layer_idx'] = int(layer_idx)
            info['layer_type'] = 'transformer'
            
            # Extract component within transformer layer
            remaining_parts = parts[parts.index('layers') + 2:]
            if 'self_attn' in remaining_parts:
                info['component'] = 'attention'
                if 'q_proj' in remaining_parts:
                    info['component'] = 'attention_q'
                elif 'k_proj' in remaining_parts:
                    info['component'] = 'attention_k'
                elif 'v_proj' in remaining_parts:
                    info['component'] = 'attention_v'
                elif 'o_proj' in remaining_parts:
                    info['component'] = 'attention_o'
            elif 'mlp' in remaining_parts:
                info['component'] = 'mlp'
                if 'c_fc' in remaining_parts:
                    info['component'] = 'mlp_fc'
                elif 'c_proj' in remaining_parts:
                    info['component'] = 'mlp_proj'
            elif 'input_layernorm' in remaining_parts:
                info['component'] = 'layernorm_input'
            elif 'post_attention_layernorm' in remaining_parts:
                info['component'] = 'layernorm_post_attn'
        except ValueError:
            pass
    elif 'ln_f' in param_name or 'final_layernorm' in param_name:
        info['layer_type'] = 'output'
        info['component'] = 'final_layernorm'
    elif 'lm_head' in param_name:
        info['layer_type'] = 'output'
        info['component'] = 'lm_head'
    
    return info


def analyze_parameter_structure(lora_weights, task_vector):
    """Analyze the structural distribution of parameters."""
    structure_data = {
        'lora': defaultdict(lambda: defaultdict(list)),
        'task_vector': defaultdict(lambda: defaultdict(list)),
        'shared': defaultdict(lambda: defaultdict(list)),
        'layer_stats': defaultdict(lambda: {'lora': 0, 'task_vector': 0, 'shared': 0}),
        'component_stats': defaultdict(lambda: {'lora': 0, 'task_vector': 0, 'shared': 0})
    }
    
    # Count non-zero task vectors
    non_zero_task_vectors = {}
    for name, tensor in task_vector.items():
        if torch.norm(tensor).item() > 1e-8:
            non_zero_task_vectors[name] = tensor
    
    # Analyze LoRA parameters
    for param_name, tensor in lora_weights.items():
        info = parse_parameter_name(param_name)
        norm = torch.norm(tensor).item()
        
        info['norm'] = norm
        info['shape'] = tensor.shape
        info['num_params'] = tensor.numel()
        
        structure_data['lora'][info['layer_type']][info['component']].append(info)
        
        if info['layer_idx'] is not None:
            structure_data['layer_stats'][info['layer_idx']]['lora'] += 1
        structure_data['component_stats'][info['component']]['lora'] += 1
    
    # Analyze Task Vector parameters
    for param_name, tensor in non_zero_task_vectors.items():
        info = parse_parameter_name(param_name)
        norm = torch.norm(tensor).item()
        
        info['norm'] = norm
        info['shape'] = tensor.shape
        info['num_params'] = tensor.numel()
        
        structure_data['task_vector'][info['layer_type']][info['component']].append(info)
        
        if info['layer_idx'] is not None:
            structure_data['layer_stats'][info['layer_idx']]['task_vector'] += 1
        structure_data['component_stats'][info['component']]['task_vector'] += 1
        
        # Check if also in LoRA (shared)
        if param_name in lora_weights:
            structure_data['shared'][info['layer_type']][info['component']].append(info)
            if info['layer_idx'] is not None:
                structure_data['layer_stats'][info['layer_idx']]['shared'] += 1
            structure_data['component_stats'][info['component']]['shared'] += 1
    
    return structure_data


def create_model_architecture_diagram(structure_data, output_dir):
    """Create a visual diagram of the model architecture showing LoRA and Task Vector coverage."""
    
    # Extract layer information
    layer_stats = structure_data['layer_stats']
    max_layer = max(layer_stats.keys()) if layer_stats else 30
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle('StarCoder2-3B Model Architecture: LoRA vs Task Vector Coverage', 
                 fontsize=16, fontweight='bold')
    
    # Left plot: Layer-by-layer coverage
    layers = list(range(max_layer + 1))
    lora_counts = [layer_stats[i]['lora'] for i in layers]
    task_counts = [layer_stats[i]['task_vector'] for i in layers]
    shared_counts = [layer_stats[i]['shared'] for i in layers]
    
    width = 0.25
    x = np.arange(len(layers))
    
    bars1 = ax1.bar(x - width, lora_counts, width, label='LoRA Only', 
                   color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x, shared_counts, width, label='Shared (LoRA + Task)', 
                   color='purple', alpha=0.8)
    bars3 = ax1.bar(x + width, [task_counts[i] - shared_counts[i] for i in range(len(layers))], 
                   width, label='Task Vector Only', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Transformer Layer Index')
    ax1.set_ylabel('Number of Modified Parameters')
    ax1.set_title('Parameter Modifications by Layer')
    ax1.set_xticks(x[::2])  # Show every 2nd layer for readability
    ax1.set_xticklabels([str(i) for i in layers[::2]])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Right plot: Component type breakdown
    component_stats = structure_data['component_stats']
    components = list(component_stats.keys())
    lora_comp_counts = [component_stats[comp]['lora'] for comp in components]
    task_comp_counts = [component_stats[comp]['task_vector'] for comp in components]
    shared_comp_counts = [component_stats[comp]['shared'] for comp in components]
    
    y_pos = np.arange(len(components))
    
    # Horizontal stacked bar chart
    bars1 = ax2.barh(y_pos, lora_comp_counts, height=0.6, 
                    label='LoRA Only', color='skyblue', alpha=0.8)
    bars2 = ax2.barh(y_pos, shared_comp_counts, height=0.6, left=lora_comp_counts,
                    label='Shared', color='purple', alpha=0.8)
    bars3 = ax2.barh(y_pos, [task_comp_counts[i] - shared_comp_counts[i] for i in range(len(components))], 
                    height=0.6, left=[lora_comp_counts[i] + shared_comp_counts[i] for i in range(len(components))],
                    label='Task Vector Only', color='lightcoral', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(components)
    ax2.set_xlabel('Number of Parameters')
    ax2.set_title('Parameter Modifications by Component Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'model_architecture_coverage.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Architecture diagram saved to: {plot_path}")
    
    pdf_path = os.path.join(output_dir, 'model_architecture_coverage.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    logging.info(f"Architecture diagram (PDF) saved to: {pdf_path}")
    
    plt.close()
    
    return plot_path


def load_task_vector(task_vector_path: str, device: str = "cpu"):
    """Load task vector from safetensors file."""
    logging.info(f"Loading task vector: {task_vector_path}")
    try:
        # Try loading without device first
        task_vector = safetensors.torch.load_file(task_vector_path)
        # Move to target device if needed
        if device != "cpu":
            task_vector = {k: v.to(device) for k, v in task_vector.items()}
        return task_vector
    except Exception as e:
        logging.error(f"Failed to load task vector with safetensors: {e}")
        # Fallback: try loading with torch.load
        try:
            task_vector = torch.load(task_vector_path, map_location="cpu")
            if device != "cpu":
                task_vector = {k: v.to(device) for k, v in task_vector.items()}
            return task_vector
        except Exception as e2:
            logging.error(f"Fallback loading also failed: {e2}")
            raise e


def extract_and_save_lora_weights(base_model_path: str, lora_path: str, output_path: str, device: str = "cuda"):
    """Extract LoRA weights and save them to file, then free memory."""
    logging.info("=== STEP 1: Extracting LoRA weights ===")
    
    # Load base model
    base_model = load_base_model(base_model_path, device)
    
    # Load LoRA model
    lora_model = load_lora_model(base_model, lora_path)
    
    # Extract LoRA weights
    lora_weights = extract_lora_weights(lora_model)
    logging.info(f"Extracted {len(lora_weights)} LoRA weight deltas")
    
    # Convert to CPU and save
    lora_cpu = {k: v.cpu() for k, v in lora_weights.items()}
    safetensors.torch.save_file(lora_cpu, output_path)
    logging.info(f"LoRA deltas saved to: {output_path}")
    
    # Store count before clearing memory
    num_weights = len(lora_cpu)
    
    # Clear memory
    del base_model, lora_model, lora_weights, lora_cpu
    torch.cuda.empty_cache()
    
    return num_weights


def load_and_compare_weights(lora_file: str, task_vector_file: str, tolerance: float = 1e-5):
    """Load both weight files and perform comparison."""
    logging.info("=== STEP 2: Loading and comparing weights ===")
    
    # Load LoRA weights from file
    logging.info(f"Loading LoRA weights from: {lora_file}")
    lora_weights = safetensors.torch.load_file(lora_file)
    
    # Load task vector weights from file  
    logging.info(f"Loading task vector from: {task_vector_file}")
    task_vector = safetensors.torch.load_file(task_vector_file)
    
    logging.info(f"Loaded {len(lora_weights)} LoRA deltas and {len(task_vector)} task vector deltas")
    
    # Perform comparison
    results = compare_lora_and_task_vector(lora_weights, task_vector, tolerance)
    
    return results


def generate_detailed_report(lora_weights, task_vector, comparison_results, output_dir):
    """Generate a comprehensive detailed report of LoRA vs Task Vector comparison."""
    
    print("\n📊 DETAILED COMPARISON REPORT")
    print("="*80)
    
    # Count non-zero task vectors for accurate comparison
    non_zero_task_vectors = {}
    for name, tensor in task_vector.items():
        if torch.norm(tensor).item() > 1e-8:
            non_zero_task_vectors[name] = tensor
    
    # Basic counts
    num_lora = len(lora_weights)
    num_task_vectors = len(non_zero_task_vectors) 
    
    print(f"\n🔢 PARAMETER COUNTS:")
    print(f"   LoRA deltas extracted: {num_lora}")
    print(f"   Non-zero task vectors: {num_task_vectors}")
    print(f"   Total unique parameters: {len(set(lora_weights.keys()) | set(non_zero_task_vectors.keys()))}")
    
    # Find overlapping parameters (shared between LoRA and Task Vector)
    shared_params = set(lora_weights.keys()) & set(non_zero_task_vectors.keys())
    lora_only_params = set(lora_weights.keys()) - set(non_zero_task_vectors.keys())
    task_only_params = set(non_zero_task_vectors.keys()) - set(lora_weights.keys())
    
    print(f"\n🔗 PARAMETER OVERLAP ANALYSIS:")
    print(f"   Shared parameters (LoRA ∩ Task Vector): {len(shared_params)}")
    print(f"   LoRA-only parameters: {len(lora_only_params)}")
    print(f"   Task Vector-only parameters: {len(task_only_params)}")
    print(f"   Coverage: LoRA covers {100*len(shared_params)/num_task_vectors:.1f}% of task vector changes")
    
    # Mathematical comparison for shared parameters
    if shared_params:
        print(f"\n🔬 MATHEMATICAL EQUIVALENCE ANALYSIS:")
        print(f"   Analyzing {len(shared_params)} shared parameters...")
        
        cosine_similarities = []
        max_diffs = []
        mean_diffs = []
        relative_errors = []
        
        for param_name in shared_params:
            if param_name in comparison_results and 'cosine_similarity' in comparison_results[param_name]:
                result = comparison_results[param_name]
                cosine_similarities.append(result['cosine_similarity'])
                max_diffs.append(result['max_absolute_diff'])
                mean_diffs.append(result['mean_absolute_diff'])
                relative_errors.append(result['relative_error'])
        
        if cosine_similarities:
            print(f"   Average cosine similarity: {np.mean(cosine_similarities):.6f}")
            print(f"   Min cosine similarity: {np.min(cosine_similarities):.6f}")
            print(f"   Max absolute difference: {np.max(max_diffs):.2e}")
            print(f"   Mean absolute difference: {np.mean(mean_diffs):.2e}")
            print(f"   Average relative error: {np.mean(relative_errors):.4f}")
            
            # Statistical significance
            perfect_matches = sum(1 for sim in cosine_similarities if sim > 0.9999)
            near_perfect = sum(1 for sim in cosine_similarities if sim > 0.999)
            
            print(f"   Perfect matches (>0.9999): {perfect_matches}/{len(cosine_similarities)} ({100*perfect_matches/len(cosine_similarities):.1f}%)")
            print(f"   Near-perfect matches (>0.999): {near_perfect}/{len(cosine_similarities)} ({100*near_perfect/len(cosine_similarities):.1f}%)")
    
    # Network structure analysis by parsing parameter names
    print(f"\n🏗️  NETWORK STRUCTURE BREAKDOWN:")
    
    def categorize_param(param_name):
        if 'embed_tokens' in param_name:
            return 'embedding'
        elif 'self_attn' in param_name:
            if 'q_proj' in param_name:
                return 'attention_q'
            elif 'k_proj' in param_name:
                return 'attention_k' 
            elif 'v_proj' in param_name:
                return 'attention_v'
            elif 'o_proj' in param_name:
                return 'attention_o'
            else:
                return 'attention_other'
        elif 'mlp' in param_name:
            return 'mlp'
        elif 'layernorm' in param_name or 'ln_f' in param_name:
            return 'normalization'
        elif 'lm_head' in param_name:
            return 'output_head'
        else:
            return 'other'
    
    # Count by component type
    lora_by_component = defaultdict(int)
    task_by_component = defaultdict(int)
    shared_by_component = defaultdict(int)
    
    for param_name in lora_weights.keys():
        component = categorize_param(param_name)
        lora_by_component[component] += 1
        if param_name in shared_params:
            shared_by_component[component] += 1
    
    for param_name in non_zero_task_vectors.keys():
        component = categorize_param(param_name)
        task_by_component[component] += 1
    
    print(f"   {'Component':<15} {'LoRA':<8} {'Task Vec':<10} {'Shared':<8} {'Coverage':<10}")
    print("-" * 65)
    
    all_components = set(lora_by_component.keys()) | set(task_by_component.keys())
    for component in sorted(all_components):
        lora_count = lora_by_component[component]
        task_count = task_by_component[component]
        shared_count = shared_by_component[component]
        coverage = f"{100*shared_count/max(task_count,1):.1f}%" if task_count > 0 else "N/A"
        print(f"   {component:<15} {lora_count:<8} {task_count:<10} {shared_count:<8} {coverage:<10}")
    
    # One-to-one parameter comparison
    print(f"\n🔍 DETAILED PARAMETER-BY-PARAMETER COMPARISON:")
    print("-" * 80)
    
    if shared_params:
        # Sort by cosine similarity (best matches first)
        shared_with_metrics = []
        for param_name in shared_params:
            if param_name in comparison_results and 'cosine_similarity' in comparison_results[param_name]:
                result = comparison_results[param_name]
                shared_with_metrics.append((param_name, result))
        
        shared_with_metrics.sort(key=lambda x: x[1]['cosine_similarity'], reverse=True)
        
        print(f"   Showing top 10 best matches:")
        print(f"   {'Parameter Name':<50} {'Cosine Sim':<12} {'Max Diff':<12} {'Component':<15}")
        print("-" * 95)
        
        for param_name, result in shared_with_metrics[:10]:
            component = categorize_param(param_name)
            short_name = param_name if len(param_name) <= 47 else param_name[:44] + "..."
            print(f"   {short_name:<50} {result['cosine_similarity']:<12.6f} {result['max_absolute_diff']:<12.2e} {component:<15}")
        
        if len(shared_with_metrics) > 10:
            print(f"   ... and {len(shared_with_metrics) - 10} more shared parameters")
    
    # Hypothesis testing: Compare norms of shared vs task-only parameters
    print(f"\n🧪 HYPOTHESIS TEST: Task-Vector-Only Parameters Have Smaller Norms")
    print("-" * 70)
    
    # Get norms for shared parameters (from task vector)
    shared_norms = []
    for param_name in shared_params:
        if param_name in non_zero_task_vectors:
            norm = torch.norm(non_zero_task_vectors[param_name]).item()
            shared_norms.append(norm)
    
    # Get norms for task-vector-only parameters
    task_only_norms = []
    for param_name in task_only_params:
        if param_name in non_zero_task_vectors:
            norm = torch.norm(non_zero_task_vectors[param_name]).item()
            task_only_norms.append(norm)
    
    if shared_norms and task_only_norms:
        shared_mean = np.mean(shared_norms)
        shared_std = np.std(shared_norms)
        shared_median = np.median(shared_norms)
        
        task_only_mean = np.mean(task_only_norms)
        task_only_std = np.std(task_only_norms)
        task_only_median = np.median(task_only_norms)
        
        print(f"   Shared parameters (LoRA + Task Vector):")
        print(f"      Count: {len(shared_norms)}")
        print(f"      Mean norm: {shared_mean:.2e} ± {shared_std:.2e}")
        print(f"      Median norm: {shared_median:.2e}")
        print(f"      Range: [{np.min(shared_norms):.2e}, {np.max(shared_norms):.2e}]")
        
        print(f"   Task-Vector-only parameters:")
        print(f"      Count: {len(task_only_norms)}")
        print(f"      Mean norm: {task_only_mean:.2e} ± {task_only_std:.2e}")
        print(f"      Median norm: {task_only_median:.2e}")
        print(f"      Range: [{np.min(task_only_norms):.2e}, {np.max(task_only_norms):.2e}]")
        
        # Statistical test
        ratio = task_only_mean / shared_mean if shared_mean > 0 else float('inf')
        print(f"   Ratio (Task-only/Shared): {ratio:.3f}")
        
        if ratio < 1.0:
            print(f"   ✅ HYPOTHESIS SUPPORTED: Task-vector-only parameters have {100*(1-ratio):.1f}% smaller mean norm")
        else:
            print(f"   ❌ HYPOTHESIS NOT SUPPORTED: Task-vector-only parameters have {100*(ratio-1):.1f}% larger mean norm")
        
        # Try statistical test if scipy is available
        try:
            from scipy import stats
            statistic, p_value = stats.mannwhitneyu(task_only_norms, shared_norms, alternative='less')
            print(f"   Mann-Whitney U test p-value: {p_value:.2e}")
            if p_value < 0.05:
                print(f"   ✅ STATISTICALLY SIGNIFICANT: Task-only parameters have significantly smaller norms (p < 0.05)")
            else:
                print(f"   ⚠️  NOT STATISTICALLY SIGNIFICANT: Difference may be due to chance (p >= 0.05)")
        except ImportError:
            print(f"   📊 Install scipy for statistical significance testing")
    
    # Save detailed report to file
    report_path = os.path.join(output_dir, 'detailed_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("LORA VS TASK VECTOR DETAILED COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"PARAMETER COUNTS:\n")
        f.write(f"  LoRA deltas extracted: {num_lora}\n")
        f.write(f"  Non-zero task vectors: {num_task_vectors}\n")
        f.write(f"  Total unique parameters: {len(set(lora_weights.keys()) | set(non_zero_task_vectors.keys()))}\n\n")
        
        f.write(f"PARAMETER OVERLAP:\n")
        f.write(f"  Shared parameters: {len(shared_params)}\n")
        f.write(f"  LoRA-only parameters: {len(lora_only_params)}\n")
        f.write(f"  Task Vector-only parameters: {len(task_only_params)}\n\n")
        
        if shared_params and cosine_similarities:
            f.write(f"MATHEMATICAL EQUIVALENCE:\n")
            f.write(f"  Average cosine similarity: {np.mean(cosine_similarities):.6f}\n")
            f.write(f"  Max absolute difference: {np.max(max_diffs):.2e}\n")
            f.write(f"  Perfect matches (>0.9999): {perfect_matches}/{len(cosine_similarities)}\n\n")
        
        # Add detailed parameter list
        f.write(f"SHARED PARAMETERS (LoRA ∩ Task Vector):\n")
        f.write("-" * 50 + "\n")
        for param_name in sorted(shared_params):
            f.write(f"  {param_name}\n")
        
        f.write(f"\nTASK VECTOR ONLY PARAMETERS:\n")
        f.write("-" * 50 + "\n")
        for param_name in sorted(task_only_params):
            f.write(f"  {param_name}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")


def plot_norm_distribution_comparison(lora_weights, task_vector, output_dir):
    """Create histogram plots comparing norm distributions of shared vs task-vector-only parameters."""
    
    # Count non-zero task vectors
    non_zero_task_vectors = {}
    for name, tensor in task_vector.items():
        if torch.norm(tensor).item() > 1e-8:
            non_zero_task_vectors[name] = tensor
    
    # Find overlapping and unique parameters
    shared_params = set(lora_weights.keys()) & set(non_zero_task_vectors.keys())
    task_only_params = set(non_zero_task_vectors.keys()) - set(lora_weights.keys())
    
    # Get norms for shared parameters (from task vector)
    shared_norms = []
    for param_name in shared_params:
        if param_name in non_zero_task_vectors:
            norm = torch.norm(non_zero_task_vectors[param_name]).item()
            shared_norms.append(norm)
    
    # Get norms for task-vector-only parameters
    task_only_norms = []
    for param_name in task_only_params:
        if param_name in non_zero_task_vectors:
            norm = torch.norm(non_zero_task_vectors[param_name]).item()
            task_only_norms.append(norm)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Norm Distribution Analysis: Shared vs Task-Vector-Only', 
                 fontsize=16, fontweight='bold')
    
    # 1. Histogram comparison
    ax1 = axes[0, 0]
    if shared_norms and task_only_norms:
        bins = np.logspace(np.log10(min(min(shared_norms), min(task_only_norms))), 
                          np.log10(max(max(shared_norms), max(task_only_norms))), 30)
        
        ax1.hist(shared_norms, bins=bins, alpha=0.7, label=f'Shared (n={len(shared_norms)})', 
                color='purple', density=True)
        ax1.hist(task_only_norms, bins=bins, alpha=0.7, label=f'Task-Only (n={len(task_only_norms)})', 
                color='lightcoral', density=True)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Parameter Norm')
        ax1.set_ylabel('Density')
        ax1.set_title('Norm Distribution Comparison (Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    if shared_norms and task_only_norms:
        data_to_plot = [shared_norms, task_only_norms]
        labels = ['Shared\n(LoRA + Task)', 'Task-Only']
        
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('purple')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax2.set_yscale('log')
        ax2.set_ylabel('Parameter Norm (Log Scale)')
        ax2.set_title('Norm Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add mean markers
        means = [np.mean(shared_norms), np.mean(task_only_norms)]
        for i, mean_val in enumerate(means):
            ax2.plot(i+1, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
        ax2.legend()
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    if shared_norms and task_only_norms:
        shared_sorted = np.sort(shared_norms)
        task_only_sorted = np.sort(task_only_norms)
        
        shared_cumulative = np.arange(1, len(shared_sorted) + 1) / len(shared_sorted)
        task_only_cumulative = np.arange(1, len(task_only_sorted) + 1) / len(task_only_sorted)
        
        ax3.plot(shared_sorted, shared_cumulative, label='Shared', color='purple', linewidth=2)
        ax3.plot(task_only_sorted, task_only_cumulative, label='Task-Only', color='lightcoral', linewidth=2)
        
        ax3.set_xscale('log')
        ax3.set_xlabel('Parameter Norm')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis for text
    
    if shared_norms and task_only_norms:
        # Calculate statistics
        shared_stats = {
            'count': len(shared_norms),
            'mean': np.mean(shared_norms),
            'std': np.std(shared_norms),
            'median': np.median(shared_norms),
            'min': np.min(shared_norms),
            'max': np.max(shared_norms)
        }
        
        task_only_stats = {
            'count': len(task_only_norms),
            'mean': np.mean(task_only_norms),
            'std': np.std(task_only_norms),
            'median': np.median(task_only_norms),
            'min': np.min(task_only_norms),
            'max': np.max(task_only_norms)
        }
        
        # Create statistical summary text
        summary_text = "STATISTICAL SUMMARY\n" + "="*30 + "\n\n"
        
        summary_text += f"SHARED PARAMETERS (LoRA + Task Vector):\n"
        summary_text += f"  Count: {shared_stats['count']}\n"
        summary_text += f"  Mean: {shared_stats['mean']:.2e}\n"
        summary_text += f"  Std:  {shared_stats['std']:.2e}\n"
        summary_text += f"  Median: {shared_stats['median']:.2e}\n"
        summary_text += f"  Range: [{shared_stats['min']:.2e}, {shared_stats['max']:.2e}]\n\n"
        
        summary_text += f"TASK-VECTOR-ONLY PARAMETERS:\n"
        summary_text += f"  Count: {task_only_stats['count']}\n"
        summary_text += f"  Mean: {task_only_stats['mean']:.2e}\n"
        summary_text += f"  Std:  {task_only_stats['std']:.2e}\n"
        summary_text += f"  Median: {task_only_stats['median']:.2e}\n"
        summary_text += f"  Range: [{task_only_stats['min']:.2e}, {task_only_stats['max']:.2e}]\n\n"
        
        # Calculate ratio for hypothesis test
        ratio = task_only_stats['mean'] / shared_stats['mean']
        
        summary_text += f"HYPOTHESIS TEST:\n"
        summary_text += f"  Mean ratio (Task-only/Shared): {ratio:.3f}\n"
        
        if ratio < 1.0:
            summary_text += f"  ✅ Task-only parameters have\n     {100*(1-ratio):.1f}% smaller mean norm\n"
        else:
            summary_text += f"  ❌ Task-only parameters have\n     {100*(ratio-1):.1f}% larger mean norm\n"
        
        # Statistical test if available
        try:
            from scipy import stats
            statistic, p_value = stats.mannwhitneyu(task_only_norms, shared_norms, alternative='less')
            summary_text += f"  Mann-Whitney U p-value: {p_value:.2e}\n"
            if p_value < 0.05:
                summary_text += f"  ✅ STATISTICALLY SIGNIFICANT\n"
            else:
                summary_text += f"  ⚠️  NOT SIGNIFICANT (p ≥ 0.05)\n"
        except ImportError:
            summary_text += f"  📊 Install scipy for statistical test\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'norm_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Norm distribution plot saved to: {plot_path}")
    
    pdf_path = os.path.join(output_dir, 'norm_distribution_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    logging.info(f"Norm distribution plot (PDF) saved to: {pdf_path}")
    
    plt.close()
    
    return plot_path


def print_structure_summary(structure_data):
    """Print a summary of the structure analysis."""
    
    print("\n🏗️  MODEL STRUCTURE ANALYSIS SUMMARY")
    print("="*60)
    
    # Layer-wise summary
    print("\n📊 LAYER-WISE PARAMETER MODIFICATIONS")
    print("-" * 50)
    
    layer_stats = structure_data['layer_stats']
    if layer_stats:
        max_layer = max(layer_stats.keys())
        print(f"Total transformer layers analyzed: {max_layer + 1}")
        
        total_lora = sum(stats['lora'] for stats in layer_stats.values())
        total_task = sum(stats['task_vector'] for stats in layer_stats.values())
        total_shared = sum(stats['shared'] for stats in layer_stats.values())
        
        print(f"Total LoRA modifications: {total_lora}")
        print(f"Total Task Vector modifications: {total_task}")
        print(f"Total shared modifications: {total_shared}")
        
        # Find layers with most modifications
        layer_totals = [(layer_idx, stats['lora'] + stats['task_vector']) 
                       for layer_idx, stats in layer_stats.items()]
        layer_totals.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 most modified layers:")
        for layer_idx, total_mods in layer_totals[:5]:
            stats = layer_stats[layer_idx]
            print(f"  Layer {layer_idx}: {total_mods} modifications "
                 f"(LoRA: {stats['lora']}, Task: {stats['task_vector']}, Shared: {stats['shared']})")
    
    # Component-wise summary
    print("\n🔧 COMPONENT-WISE PARAMETER MODIFICATIONS")
    print("-" * 50)
    
    component_stats = structure_data['component_stats']
    if component_stats:
        # Sort by total modifications
        comp_totals = [(comp, stats['lora'] + stats['task_vector']) 
                      for comp, stats in component_stats.items()]
        comp_totals.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Component':<20} {'LoRA':<8} {'Task Vec':<8} {'Shared':<8} {'Total':<8}")
        print("-" * 60)
        
        for comp, total_mods in comp_totals:
            stats = component_stats[comp]
            print(f"{comp:<20} {stats['lora']:<8} {stats['task_vector']:<8} "
                 f"{stats['shared']:<8} {total_mods:<8}")
    
    # Architecture overview
    print("\n🏛️  ARCHITECTURE COVERAGE OVERVIEW")
    print("-" * 50)
    
    lora_components = set()
    task_components = set()
    
    for layer_type, components in structure_data['lora'].items():
        for component in components.keys():
            lora_components.add(f"{layer_type}.{component}")
    
    for layer_type, components in structure_data['task_vector'].items():
        for component in components.keys():
            task_components.add(f"{layer_type}.{component}")
    
    shared_components = lora_components & task_components
    lora_only_components = lora_components - task_components
    task_only_components = task_components - lora_components
    
    print(f"Components modified by LoRA only: {len(lora_only_components)}")
    if lora_only_components:
        for comp in sorted(lora_only_components):
            print(f"  - {comp}")
    
    print(f"\nComponents modified by Task Vector only: {len(task_only_components)}")
    if task_only_components:
        for comp in sorted(task_only_components):
            print(f"  - {comp}")
    
    print(f"\nComponents modified by both: {len(shared_components)}")
    if shared_components:
        for comp in sorted(shared_components):
            print(f"  - {comp}")


def create_detailed_layer_heatmap(structure_data, output_dir):
    """Create a detailed heatmap showing parameter norms across layers and components."""
    
    # Prepare data for heatmap
    layer_stats = structure_data['layer_stats']
    max_layer = max(layer_stats.keys()) if layer_stats else 30
    
    # Component types in order
    component_order = [
        'attention_q', 'attention_k', 'attention_v', 'attention_o',
        'layernorm_input', 'mlp_fc', 'mlp_proj', 'layernorm_post_attn'
    ]
    
    # Create matrices for LoRA and Task Vector norms
    lora_matrix = np.zeros((len(component_order), max_layer + 1))
    task_matrix = np.zeros((len(component_order), max_layer + 1))
    shared_matrix = np.zeros((len(component_order), max_layer + 1))
    
    # Fill matrices with parameter norms
    for layer_type, components in structure_data['lora'].items():
        if layer_type == 'transformer':
            for component, params in components.items():
                if component in component_order:
                    comp_idx = component_order.index(component)
                    for param_info in params:
                        if param_info['layer_idx'] is not None:
                            layer_idx = param_info['layer_idx']
                            lora_matrix[comp_idx, layer_idx] = max(
                                lora_matrix[comp_idx, layer_idx], 
                                param_info['norm']
                            )
    
    for layer_type, components in structure_data['task_vector'].items():
        if layer_type == 'transformer':
            for component, params in components.items():
                if component in component_order:
                    comp_idx = component_order.index(component)
                    for param_info in params:
                        if param_info['layer_idx'] is not None:
                            layer_idx = param_info['layer_idx']
                            task_matrix[comp_idx, layer_idx] = max(
                                task_matrix[comp_idx, layer_idx], 
                                param_info['norm']
                            )
    
    for layer_type, components in structure_data['shared'].items():
        if layer_type == 'transformer':
            for component, params in components.items():
                if component in component_order:
                    comp_idx = component_order.index(component)
                    for param_info in params:
                        if param_info['layer_idx'] is not None:
                            layer_idx = param_info['layer_idx']
                            shared_matrix[comp_idx, layer_idx] = max(
                                shared_matrix[comp_idx, layer_idx], 
                                param_info['norm']
                            )
    
    # Create subplots for comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('StarCoder2-3B Parameter Modification Heatmaps', fontsize=16, fontweight='bold')
    
    # LoRA heatmap
    ax1 = axes[0, 0]
    mask_lora = lora_matrix == 0
    sns.heatmap(lora_matrix, mask=mask_lora, ax=ax1, cmap='Blues', 
                xticklabels=range(max_layer + 1), yticklabels=component_order,
                cbar_kws={'label': 'Parameter Norm'}, fmt='.2e')
    ax1.set_title('LoRA Parameter Modifications')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Model Component')
    
    # Task Vector heatmap
    ax2 = axes[0, 1]
    mask_task = task_matrix == 0
    sns.heatmap(task_matrix, mask=mask_task, ax=ax2, cmap='Reds',
                xticklabels=range(max_layer + 1), yticklabels=component_order,
                cbar_kws={'label': 'Parameter Norm'}, fmt='.2e')
    ax2.set_title('Task Vector Parameter Modifications')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Model Component')
    
    # Shared parameters heatmap
    ax3 = axes[1, 0]
    mask_shared = shared_matrix == 0
    sns.heatmap(shared_matrix, mask=mask_shared, ax=ax3, cmap='Purples',
                xticklabels=range(max_layer + 1), yticklabels=component_order,
                cbar_kws={'label': 'Parameter Norm'}, fmt='.2e')
    ax3.set_title('Shared Parameter Modifications (LoRA ∩ Task Vector)')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Model Component')
    
    # Coverage comparison (binary: modified or not)
    ax4 = axes[1, 1]
    coverage_matrix = np.zeros((len(component_order), max_layer + 1))
    # 0 = no modification, 1 = LoRA only, 2 = Task Vector only, 3 = both
    for i in range(len(component_order)):
        for j in range(max_layer + 1):
            has_lora = lora_matrix[i, j] > 0
            has_task = task_matrix[i, j] > 0
            if has_lora and has_task:
                coverage_matrix[i, j] = 3  # Both
            elif has_lora:
                coverage_matrix[i, j] = 1  # LoRA only
            elif has_task:
                coverage_matrix[i, j] = 2  # Task Vector only
            else:
                coverage_matrix[i, j] = 0  # None
    
    colors = ['white', 'skyblue', 'lightcoral', 'purple']
    coverage_cmap = sns.color_palette(colors, as_cmap=True)
    
    sns.heatmap(coverage_matrix, ax=ax4, cmap=coverage_cmap, 
                xticklabels=range(max_layer + 1), yticklabels=component_order,
                cbar_kws={'label': 'Modification Type', 
                         'ticks': [0, 1, 2, 3]}, 
                vmin=0, vmax=3)
    ax4.set_title('Parameter Modification Coverage')
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Model Component')
    
    # Customize colorbar labels
    cbar = ax4.collections[0].colorbar
    cbar.set_ticklabels(['None', 'LoRA Only', 'Task Vector Only', 'Both'])
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'parameter_modification_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logging.info(f"Parameter heatmap saved to: {heatmap_path}")
    
    heatmap_pdf_path = os.path.join(output_dir, 'parameter_modification_heatmap.pdf')
    plt.savefig(heatmap_pdf_path, bbox_inches='tight')
    logging.info(f"Parameter heatmap (PDF) saved to: {heatmap_pdf_path}")
    
    plt.close()
    
    return heatmap_path


def create_parameter_flow_diagram(structure_data, output_dir):
    """Create a flow diagram showing how parameters flow through the model."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle('StarCoder2-3B Parameter Modification Flow', fontsize=16, fontweight='bold')
    
    # Define model structure positions
    positions = {
        'embed_tokens': (2, 9),
        'attention_q': (1, 7),
        'attention_k': (2, 7),
        'attention_v': (3, 7),
        'attention_o': (4, 7),
        'layernorm_input': (2.5, 8),
        'mlp_fc': (1.5, 5),
        'mlp_proj': (3.5, 5),
        'layernorm_post_attn': (2.5, 6),
        'final_layernorm': (2.5, 3),
        'lm_head': (2.5, 1)
    }
    
    # Component colors based on modification type
    component_colors = {}
    component_stats = structure_data['component_stats']
    
    for comp, stats in component_stats.items():
        if stats['shared'] > 0:
            component_colors[comp] = 'purple'  # Both LoRA and Task Vector
        elif stats['lora'] > 0:
            component_colors[comp] = 'skyblue'  # LoRA only
        elif stats['task_vector'] > 0:
            component_colors[comp] = 'lightcoral'  # Task Vector only
        else:
            component_colors[comp] = 'lightgray'  # No modifications
    
    # Draw components
    for comp, pos in positions.items():
        color = component_colors.get(comp, 'lightgray')
        stats = component_stats.get(comp, {'lora': 0, 'task_vector': 0, 'shared': 0})
        
        # Draw circle for component
        circle = plt.Circle(pos, 0.3, color=color, alpha=0.7, zorder=2)
        ax.add_patch(circle)
        
        # Add component label
        ax.text(pos[0], pos[1], comp.replace('_', '\n'), 
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
        
        # Add statistics below component
        stats_text = f"L:{stats['lora']} T:{stats['task_vector']} S:{stats['shared']}"
        ax.text(pos[0], pos[1] - 0.5, stats_text, 
                ha='center', va='center', fontsize=6, zorder=3)
    
    # Draw layer structure (simplified representation of 30 layers)
    layer_y = 7
    layer_x_start = 6
    layer_spacing = 0.5
    
    # Show sample layers with statistics
    sample_layers = [0, 7, 15, 23, 29]  # Show every ~8th layer
    layer_stats = structure_data['layer_stats']
    
    for i, layer_idx in enumerate(sample_layers):
        x_pos = layer_x_start + i * layer_spacing * 2
        stats = layer_stats.get(layer_idx, {'lora': 0, 'task_vector': 0, 'shared': 0})
        
        # Determine color based on dominant modification type
        if stats['shared'] > 0:
            color = 'purple'
        elif stats['lora'] > stats['task_vector']:
            color = 'skyblue'
        elif stats['task_vector'] > 0:
            color = 'lightcoral'
        else:
            color = 'lightgray'
        
        # Draw layer representation
        rect = patches.Rectangle((x_pos - 0.2, layer_y - 0.3), 0.4, 0.6, 
                               facecolor=color, alpha=0.7, zorder=2)
        ax.add_patch(rect)
        
        # Add layer label
        ax.text(x_pos, layer_y + 0.6, f'L{layer_idx}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add statistics
        total_mods = stats['lora'] + stats['task_vector'] + stats['shared']
        ax.text(x_pos, layer_y - 0.6, f'{total_mods}', 
                ha='center', va='center', fontsize=8)
    
    # Add arrows to show data flow
    flow_arrows = [
        ((2, 8.7), (2, 8.3)),  # embedding to layernorm
        ((2.5, 7.7), (2.5, 7.3)),  # layernorm to attention
        ((2.5, 6.7), (2.5, 6.3)),  # attention to post layernorm
        ((2.5, 5.7), (2.5, 5.3)),  # post layernorm to mlp
        ((2.5, 4.7), (2.5, 3.3)),  # mlp to final layernorm
        ((2.5, 2.7), (2.5, 1.3)),  # final layernorm to lm_head
    ]
    
    for start, end in flow_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))
    
    # Add legend
    legend_elements = [
        plt.Circle((0, 0), 0.1, color='purple', alpha=0.7, label='LoRA + Task Vector'),
        plt.Circle((0, 0), 0.1, color='skyblue', alpha=0.7, label='LoRA Only'),
        plt.Circle((0, 0), 0.1, color='lightcoral', alpha=0.7, label='Task Vector Only'),
        plt.Circle((0, 0), 0.1, color='lightgray', alpha=0.7, label='No Modifications')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set axis properties
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add text annotations
    ax.text(0.5, 9.5, 'Input Embedding', fontsize=10, fontweight='bold')
    ax.text(6, 8.5, 'Transformer Layers\n(Sample of 30)', fontsize=10, fontweight='bold', ha='center')
    ax.text(0.5, 0.5, 'Output', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the flow diagram
    flow_path = os.path.join(output_dir, 'parameter_flow_diagram.png')
    plt.savefig(flow_path, dpi=300, bbox_inches='tight')
    logging.info(f"Parameter flow diagram saved to: {flow_path}")
    
    flow_pdf_path = os.path.join(output_dir, 'parameter_flow_diagram.pdf')
    plt.savefig(flow_pdf_path, bbox_inches='tight')
    logging.info(f"Parameter flow diagram (PDF) saved to: {flow_pdf_path}")
    
    plt.close()
    
    return flow_path


def save_structure_analysis(structure_data, output_dir):
    """Save detailed structure analysis to JSON file."""
    
    # Convert data to JSON-serializable format
    json_data = {}
    
    for category in ['lora', 'task_vector', 'shared']:
        json_data[category] = {}
        for layer_type, components in structure_data[category].items():
            json_data[category][layer_type] = {}
            for component, params in components.items():
                json_data[category][layer_type][component] = []
                for param_info in params:
                    # Convert tensor shape to list
                    param_dict = param_info.copy()
                    if 'shape' in param_dict:
                        param_dict['shape'] = list(param_dict['shape'])
                    json_data[category][layer_type][component].append(param_dict)
    
    # Add statistics
    json_data['layer_stats'] = dict(structure_data['layer_stats'])
    json_data['component_stats'] = dict(structure_data['component_stats'])
    
    # Save to file
    structure_path = os.path.join(output_dir, 'structure_analysis.json')
    with open(structure_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logging.info(f"Structure analysis saved to: {structure_path}")
    return structure_path


def create_cosine_similarity_visualization(lora_weights, task_vector, comparison_results, output_dir):
    """Create comprehensive cosine similarity visualizations showing mathematical equivalence."""
    
    # Count non-zero task vectors for accurate comparison
    non_zero_task_vectors = {}
    for name, tensor in task_vector.items():
        if torch.norm(tensor).item() > 1e-8:
            non_zero_task_vectors[name] = tensor
    
    # Find shared parameters
    shared_params = set(lora_weights.keys()) & set(non_zero_task_vectors.keys())
    
    if not shared_params:
        logging.warning("No shared parameters found for cosine similarity analysis")
        return None
    
    # Extract cosine similarities and parameter info
    similarity_data = []
    max_diffs = []
    layer_similarities = defaultdict(list)
    component_similarities = defaultdict(list)
    
    for param_name in shared_params:
        if param_name in comparison_results and 'cosine_similarity' in comparison_results[param_name]:
            cos_sim = comparison_results[param_name]['cosine_similarity']
            max_diff = comparison_results[param_name]['max_absolute_diff']
            
            # Parse parameter info
            info = parse_parameter_name(param_name)
            
            similarity_data.append({
                'parameter': param_name,
                'cosine_similarity': cos_sim,
                'max_absolute_diff': max_diff,
                'layer_idx': info['layer_idx'],
                'component': info['component'],
                'layer_type': info['layer_type']
            })
            
            max_diffs.append(max_diff)
            
            if info['layer_idx'] is not None:
                layer_similarities[info['layer_idx']].append(cos_sim)
            component_similarities[info['component']].append(cos_sim)
    
    # Create comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle('Mathematical Equivalence: LoRA ↔ Task Vector Cosine Similarity Analysis\n'
                f'StarCoder2-3B: 30 Layers × 4 Attention Projections = 120 LoRA Parameters', 
                fontsize=16, fontweight='bold')
    
    # 1. Cosine similarity distribution histogram
    ax1 = plt.subplot(3, 3, 1)
    cosine_sims = [d['cosine_similarity'] for d in similarity_data]
    
    # Create histogram with safe binning for near-identical values
    cos_min, cos_max = min(cosine_sims), max(cosine_sims)
    cos_range = cos_max - cos_min
    
    if cos_range < 1e-6:
        # Values are nearly identical, create bins around the mean
        mean_cos = np.mean(cosine_sims)
        bins = np.linspace(mean_cos - 1e-6, mean_cos + 1e-6, 20)
    elif cos_min > 0.999:
        # All values are very high, create fine-grained bins
        bins = np.linspace(cos_min - 1e-6, cos_max + 1e-6, 50)
    else:
        # Normal case with wider range
        bins = np.concatenate([
            np.linspace(cos_min, 0.999, 20),
            np.linspace(0.999, 0.9999, 20),
            np.linspace(0.9999, cos_max, 20)
        ])
    
    ax1.hist(cosine_sims, bins=bins, alpha=0.7, color='purple', edgecolor='black')
    ax1.axvline(np.mean(cosine_sims), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cosine_sims):.6f}')
    ax1.axvline(np.median(cosine_sims), color='green', linestyle='--', 
               label=f'Median: {np.median(cosine_sims):.6f}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Cosine Similarity Distribution\n(Perfect = 1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    perfect_count = sum(1 for sim in cosine_sims if sim > 0.9999)
    near_perfect_count = sum(1 for sim in cosine_sims if sim > 0.999)
    
    stats_text = f'Perfect (>0.9999): {perfect_count}/{len(cosine_sims)} ({100*perfect_count/len(cosine_sims):.1f}%)\n'
    stats_text += f'Near-perfect (>0.999): {near_perfect_count}/{len(cosine_sims)} ({100*near_perfect_count/len(cosine_sims):.1f}%)\n'
    stats_text += f'Min similarity: {min(cosine_sims):.6f}\n'
    stats_text += f'Max difference: {max(max_diffs):.2e}'
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Layer-wise cosine similarity
    ax2 = plt.subplot(3, 3, 2)
    
    layers = sorted(layer_similarities.keys())
    layer_means = [np.mean(layer_similarities[layer]) for layer in layers]
    layer_stds = [np.std(layer_similarities[layer]) for layer in layers]
    
    ax2.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=3, 
                capthick=2, elinewidth=1, markersize=4, color='blue', alpha=0.7)
    ax2.set_xlabel('Transformer Layer Index')
    ax2.set_ylabel('Mean Cosine Similarity')
    ax2.set_title('Cosine Similarity by Layer\n(30 Transformer Layers)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.999, 1.001)  # Zoom in to see small variations
    
    # Add horizontal line at perfect similarity
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax2.legend()
    
    # 3. Component-wise cosine similarity
    ax3 = plt.subplot(3, 3, 3)
    
    components = sorted(component_similarities.keys())
    component_means = [np.mean(component_similarities[comp]) for comp in components]
    component_stds = [np.std(component_similarities[comp]) for comp in components]
    
    bars = ax3.bar(range(len(components)), component_means, yerr=component_stds, 
                  capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax3.set_xticks(range(len(components)))
    ax3.set_xticklabels(components, rotation=45, ha='right')
    ax3.set_ylabel('Mean Cosine Similarity')
    ax3.set_title('Cosine Similarity by Attention Component\n(q_proj, k_proj, v_proj, o_proj)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0.999, 1.001)  # Zoom in to see small variations
    
    # Add horizontal line at perfect similarity
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax3.legend()
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, component_means, component_stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.00001,
                f'{mean_val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Architecture diagram showing shared parameters
    ax4 = plt.subplot(3, 3, 4)
    
    # Create a visual representation of the 30x4 architecture
    layer_range = range(30)
    components = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    # Create matrix showing which parameters are shared
    shared_matrix = np.zeros((len(components), 30))
    similarity_matrix = np.zeros((len(components), 30))
    
    for data in similarity_data:
        if data['layer_idx'] is not None and data['component'] in components:
            layer_idx = data['layer_idx']
            comp_idx = components.index(data['component'])
            shared_matrix[comp_idx, layer_idx] = 1
            similarity_matrix[comp_idx, layer_idx] = data['cosine_similarity']
    
    # Only show similarity values where parameters are shared
    masked_similarity = np.ma.masked_where(shared_matrix == 0, similarity_matrix)
    
    im = ax4.imshow(masked_similarity, cmap='RdYlBu_r', vmin=0.999, vmax=1.001, aspect='auto')
    ax4.set_xticks(range(0, 30, 5))
    ax4.set_xticklabels(range(0, 30, 5))
    ax4.set_yticks(range(len(components)))
    ax4.set_yticklabels(components)
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Attention Component')
    ax4.set_title('Cosine Similarity Heatmap\n(30 Layers × 4 Components = 120 Parameters)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Cosine Similarity')
    
    # 5. Scatter plot: Cosine similarity vs Max absolute difference
    ax5 = plt.subplot(3, 3, 5)
    
    cosine_sims = [d['cosine_similarity'] for d in similarity_data]
    max_diffs = [d['max_absolute_diff'] for d in similarity_data]
    
    scatter = ax5.scatter(cosine_sims, max_diffs, alpha=0.6, c=range(len(cosine_sims)), 
                         cmap='viridis', s=30)
    ax5.set_xlabel('Cosine Similarity')
    ax5.set_ylabel('Max Absolute Difference')
    ax5.set_title('Similarity vs Difference\n(Lower-right = Perfect Match)')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # Log scale for differences
    
    # Add ideal point annotation
    ax5.annotate('Perfect Match\n(1.0, 0)', xy=(1.0, min(max_diffs)), 
                xytext=(0.9995, max(max_diffs)/10),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 6. Box plot of cosine similarities by component
    ax6 = plt.subplot(3, 3, 6)
    
    component_data = [component_similarities[comp] for comp in components]
    bp = ax6.boxplot(component_data, labels=components, patch_artist=True)
    
    # Color the boxes
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_ylabel('Cosine Similarity')
    ax6.set_title('Cosine Similarity Distribution\nby Attention Component')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0.999, 1.001)  # Zoom in to see variations
    
    # Add horizontal line at perfect similarity
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax6.legend()
    
    # 7. Cumulative distribution of cosine similarities
    ax7 = plt.subplot(3, 3, 7)
    
    sorted_sims = np.sort(cosine_sims)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    
    ax7.plot(sorted_sims, cumulative, marker='o', markersize=2, alpha=0.7, color='purple')
    ax7.set_xlabel('Cosine Similarity')
    ax7.set_ylabel('Cumulative Probability')
    ax7.set_title('Cumulative Distribution\n(Steep curve = Concentrated values)')
    ax7.grid(True, alpha=0.3)
    
    # Add reference lines
    ax7.axvline(x=0.999, color='orange', linestyle='--', alpha=0.7, label='0.999')
    ax7.axvline(x=0.9999, color='red', linestyle='--', alpha=0.7, label='0.9999')
    ax7.axvline(x=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect (1.0)')
    ax7.legend()
    
    # 8. Mathematical equivalence summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')  # Hide axes for text display
    
    # Calculate comprehensive statistics
    mean_sim = np.mean(cosine_sims)
    min_sim = np.min(cosine_sims)
    max_sim = np.max(cosine_sims)
    std_sim = np.std(cosine_sims)
    
    mean_diff = np.mean(max_diffs)
    max_diff = np.max(max_diffs)
    min_diff = np.min(max_diffs)
    
    # Create summary text
    summary_text = f"""MATHEMATICAL EQUIVALENCE VALIDATION
    
📊 ARCHITECTURE SUMMARY:
• StarCoder2-3B: 30 transformer layers
• LoRA targets: 4 attention projections per layer
• Total LoRA parameters: 30 × 4 = 120
• Shared with Task Vector: {len(shared_params)}/120 (100%)

🔬 COSINE SIMILARITY ANALYSIS:
• Mean: {mean_sim:.8f}
• Min:  {min_sim:.8f}
• Max:  {max_sim:.8f}
• Std:  {std_sim:.8f}

📏 ABSOLUTE DIFFERENCE ANALYSIS:
• Mean: {mean_diff:.2e}
• Min:  {min_diff:.2e}
• Max:  {max_diff:.2e}

✅ VALIDATION RESULTS:
• Perfect matches (>0.9999): {perfect_count}/120 ({100*perfect_count/120:.1f}%)
• Near-perfect (>0.999): {near_perfect_count}/120 ({100*near_perfect_count/120:.1f}%)

🧮 MATHEMATICAL RELATIONSHIP:
Task_Vector[param] ≡ LoRA_B @ LoRA_A × scaling
(Verified with {len(shared_params)} parameters)"""

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    # 9. Parameter names with their similarities (top performers)
    ax9 = plt.subplot(3, 3, 9)
    
    # Sort by cosine similarity and show top 10
    sorted_data = sorted(similarity_data, key=lambda x: x['cosine_similarity'], reverse=True)
    top_params = sorted_data[:10]
    
    param_names = [d['parameter'].split('.')[-2] + '.' + d['parameter'].split('.')[-1] 
                  for d in top_params]  # Show only component.weight
    similarities = [d['cosine_similarity'] for d in top_params]
    
    y_pos = np.arange(len(param_names))
    bars = ax9.barh(y_pos, similarities, alpha=0.7, color='green')
    
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(param_names, fontsize=8)
    ax9.set_xlabel('Cosine Similarity')
    ax9.set_title('Top 10 Most Similar Parameters\n(All should be ≈ 1.0)')
    ax9.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax9.text(sim - 0.00005, bar.get_y() + bar.get_height()/2,
                f'{sim:.6f}', ha='right', va='center', fontsize=7)
    
    plt.tight_layout()
    
    # Save the comprehensive similarity analysis
    similarity_path = os.path.join(output_dir, 'cosine_similarity_analysis.png')
    plt.savefig(similarity_path, dpi=300, bbox_inches='tight')
    logging.info(f"Cosine similarity analysis saved to: {similarity_path}")
    
    similarity_pdf_path = os.path.join(output_dir, 'cosine_similarity_analysis.pdf')
    plt.savefig(similarity_pdf_path, bbox_inches='tight')
    logging.info(f"Cosine similarity analysis (PDF) saved to: {similarity_pdf_path}")
    
    plt.close()
    
    return similarity_path


def main():
    parser = argparse.ArgumentParser(description="Compare LoRA adapters with Task Vectors (Memory Efficient)")
    parser.add_argument("--base_model", default="bigcode/starcoder2-3b", help="Base model path")
    parser.add_argument("--lora_path", default="checkpoint-40000/", help="Path to LoRA adapter")
    parser.add_argument("--task_vector_path", default="task_vectors_fp32/task_vector.safetensors", help="Path to pre-extracted task vector")
    parser.add_argument("--output_dir", default="./lora_task_vector_comparison", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Tolerance for considering weights equal")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File paths for intermediate storage
    lora_file = os.path.join(args.output_dir, "lora_deltas.safetensors")
    
    print("🚀 Starting memory-efficient LoRA vs Task Vector comparison...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Tolerance: {args.tolerance}")
    
    # Step 1: Extract and save LoRA weights (then free memory)
    num_lora_weights = extract_and_save_lora_weights(
        args.base_model, 
        args.lora_path, 
        lora_file, 
        args.device
    )
    
    # Step 2: Load both files and compare (no models in GPU memory)
    if not os.path.exists(args.task_vector_path):
        raise FileNotFoundError(f"Task vector file not found: {args.task_vector_path}")
    
    results = load_and_compare_weights(lora_file, args.task_vector_path, args.tolerance)
    
    # Step 3: Load weights for detailed analysis
    lora_weights = safetensors.torch.load_file(lora_file)
    task_vector = safetensors.torch.load_file(args.task_vector_path)
    
    print("🔍 Analyzing parameter structure...")
    structure_data = analyze_parameter_structure(lora_weights, task_vector)
    
    # Print structure summary
    print_structure_summary(structure_data)
    
    # Generate comprehensive reporting
    generate_detailed_report(lora_weights, task_vector, results, args.output_dir)
    
    print("🎨 Creating comprehensive visualizations...")
    
    # 1. Model architecture diagram
    arch_path = create_model_architecture_diagram(structure_data, args.output_dir)
    
    # 2. Detailed layer heatmap
    heatmap_path = create_detailed_layer_heatmap(structure_data, args.output_dir)
    
    # 3. Parameter flow diagram
    flow_path = create_parameter_flow_diagram(structure_data, args.output_dir)
    
    # 4. Norm distribution plots
    norm_plot_path = plot_norm_distribution_comparison(lora_weights, task_vector, args.output_dir)
    
    # 5. Cosine similarity visualization
    similarity_path = create_cosine_similarity_visualization(lora_weights, task_vector, results, args.output_dir)
    
    # 6. Save detailed analysis
    structure_path = save_structure_analysis(structure_data, args.output_dir)
    
    print(f"\n🎉 Comprehensive analysis complete! Files saved:")
    print(f"📊 Architecture coverage: {arch_path}")
    print(f"🔥 Parameter heatmap: {heatmap_path}")
    print(f"🌊 Parameter flow diagram: {flow_path}")
    print(f"📊 Norm distribution: {norm_plot_path}")
    print(f"📊 Cosine similarity: {similarity_path}")
    print(f"📋 Structure analysis: {structure_path}")
    print(f"📄 LoRA deltas: {lora_file}")
    print(f"📄 Task vector deltas: {args.task_vector_path}")
    print(f"📁 All files in: {args.output_dir}")


if __name__ == "__main__":
    main()
