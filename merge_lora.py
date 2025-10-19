#!/usr/bin/env python3
"""
LoRA Model Merger

This script merges a base model with LoRA adapters to create a fine-tuned model.
Supports both FP16 and FP32 precision with configurable parameters.

Usage:
    python merge_lora.py --base_model bigcode/starcoder2-3b --lora_path checkpoint-40000/ --output_dir merged_model_fp32 --precision fp32
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_model(
    base_model_name: str,
    lora_path: str, 
    output_dir: str,
    precision: str = "fp32",
    device: str = "cuda:0"
):
    """
    Merge base model with LoRA adapter.
    
    Args:
        base_model_name: HuggingFace model name (e.g., 'bigcode/starcoder2-3b')
        lora_path: Path to LoRA checkpoint directory
        output_dir: Directory to save merged model
        precision: Model precision ('fp16' or 'fp32')
        device: Device to load model on
    """
    
    # Set torch dtype based on precision
    if precision.lower() == "fp16":
        torch_dtype = torch.float16
        print(f"🔧 Using FP16 precision (torch.float16)")
    elif precision.lower() == "fp32":
        torch_dtype = torch.float32
        print(f"🔧 Using FP32 precision (torch.float32)")
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'fp32'")
    
    print(f"📥 Loading base model: {base_model_name}")
    print(f"📍 Target device: {device}")
    print(f"📁 LoRA path: {lora_path}")
    print(f"💾 Output directory: {output_dir}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        torch_dtype=torch_dtype
    )
    
    print(f"🔗 Loading LoRA adapter from: {lora_path}")
    # Load LoRA model
    model = PeftModel.from_pretrained(model, lora_path)
    
    print(f"🔄 Merging LoRA adapter with base model...")
    # Merge and unload LoRA
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Saving merged model to: {output_dir}")
    # Save merged model
    model.save_pretrained(output_dir)
    
    # Load and save tokenizer
    print(f"📝 Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model merging completed successfully!")
    print(f"📊 Model precision: {precision.upper()}")
    print(f"📁 Merged model saved to: {os.path.abspath(output_dir)}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    
    parser.add_argument("--base_model", 
                       default="bigcode/starcoder2-3b",
                       help="Base model name from HuggingFace")
    
    parser.add_argument("--lora_path", 
                       default="checkpoint-40000/",
                       help="Path to LoRA checkpoint directory")
    
    parser.add_argument("--output_dir", 
                       default="merged_model",
                       help="Output directory for merged model")
    
    parser.add_argument("--precision", 
                       choices=["fp16", "fp32"],
                       default="fp32",
                       help="Model precision (fp16 or fp32). IMPORTANT: fp32 prevents precision artifacts in task vectors!")
    
    parser.add_argument("--device", 
                       default="cuda:0",
                       help="Device to load model on")
    
    args = parser.parse_args()
    
    print("🚀 LoRA Model Merger")
    print("=" * 50)
    print(f"⚠️  PRECISION NOTE: Using {args.precision.upper()} precision")
    if args.precision == "fp16":
        print("   ⚠️  WARNING: FP16 may introduce precision artifacts in task vectors!")
        print("   💡 TIP: Use --precision fp32 for mathematical precision")
    else:
        print("   ✅ FP32 provides full mathematical precision for task vector analysis")
    print()
    
    # Perform the merge
    output_path = merge_lora_model(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        precision=args.precision,
        device=args.device
    )
    
    print("\n📋 Next Steps:")
    print(f"1. Extract task vectors using the merged model at: {output_path}")
    print(f"2. Compare LoRA deltas with task vectors for mathematical analysis")
    print(f"3. Example task vector extraction:")
    print(f"   python extract_task_vectors.py --finetuned_model {output_path} --output_dir task_vectors_{args.precision}")


if __name__ == "__main__":
    main()


