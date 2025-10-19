# Setup Guide

This guide will help you set up the environment and run the LoRA vs Task Vector analysis.

## 🔧 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for large models)
- At least 16GB RAM (32GB recommended for 3B models)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lora-vs-task-vectors.git
cd lora-vs-task-vectors/custom_script
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv lora-analysis-env
source lora-analysis-env/bin/activate  # On Windows: lora-analysis-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch, transformers, safetensors, peft; print('All dependencies installed successfully!')"
```

## 🚀 Quick Start

### Option 1: Run Complete Analysis
Run our complete validation on StarCoder2-3B:
```bash
python final_validation.py
```

### Option 2: Analyze Your Own Models
```bash
# Step 1: Merge your LoRA adapter with base model
python merge_lora.py \
  --base_model your_base_model_name \
  --lora_path /path/to/your/lora/adapter \
  --output_path /path/to/merged/model

# Step 2: Extract task vectors
python extract_task_vectors.py \
  --base_model your_base_model_name \
  --merged_model /path/to/merged/model \
  --output_path /path/to/task/vectors

# Step 3: Compare LoRA vs Task Vector
python compare_lora_task_vector.py \
  --lora_path checkpoint-40000/ \
  --task_vector_path /path/to/task/vectors/task_vector.safetensors
```

## 📁 File Structure Overview

```
custom_script/
├── 🔧 Core Analysis Tools
│   ├── final_validation.py          # Complete validation pipeline
│   ├── compare_lora_task_vector.py  # Main comparison tool
│   └── merge_lora.py                # LoRA merging (precision-fixed)
│
├── 📊 Analysis Scripts
│   ├── analyze_lora_vs_task_vector_math.py  # Mathematical validation
│   ├── simple_task_vector_analysis.py      # Architecture analysis
│   └── investigate_parameter_changes.py    # Parameter investigation
│
├── 🛠️ Utility Scripts
│   ├── extract_task_vectors.py      # Task vector extraction
│   ├── quick_precision_test.py      # Precision testing
│   └── visualize_model_structure.py # Model visualization
│
└── 📚 Documentation
    ├── README.md                           # Main documentation
    ├── STARCODER2_ARCHITECTURE_ANALYSIS.md # Architecture details
    └── EXPERIMENTAL_VALIDATION_SUMMARY.md  # Results summary
```

## 🎯 Usage Examples

### Example 1: Basic Parameter Analysis
```bash
# Analyze task vector structure
python simple_task_vector_analysis.py
```

### Example 2: Precision Impact Testing
```bash
# Test float16 vs float32 impact
python quick_precision_test.py
```

### Example 3: Mathematical Validation
```bash
# Run mathematical equivalence tests
python analyze_lora_vs_task_vector_math.py
```

## 🔍 Understanding the Output

### Mathematical Validation Results
- **Cosine Similarity**: Should be ≈ 1.0 for equivalent vectors
- **Mean Absolute Error**: Should be < 1e-6 for numerical precision
- **Max Absolute Error**: Should be < 1e-4 for practical equivalence

### Parameter Analysis Results
- **Total Parameters**: Complete model architecture count
- **Modified Parameters**: Parameters changed by LoRA
- **Zero Parameters**: Unchanged parameters in task vector

### Architecture Breakdown
- **Attention Parameters**: Modified by LoRA (weights only)
- **MLP Parameters**: Unchanged by LoRA
- **LayerNorm Parameters**: Unchanged by LoRA
- **Embedding Parameters**: Unchanged by LoRA

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Solution: Use CPU or smaller batch sizes
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### Missing Model Files
```bash
# Solution: Download models first
huggingface-cli download bigcode/starcoder2-3b
```

#### Precision Warnings
```bash
# Solution: Ensure float32 usage in merge scripts
# Check merge_lora.py uses torch_dtype=torch.float32
```

### Performance Tips

#### Memory Optimization
- Use CPU for analysis if GPU memory is limited
- Process models in smaller chunks if needed
- Clear GPU cache between operations: `torch.cuda.empty_cache()`

#### Speed Optimization
- Use GPU for tensor operations when possible
- Cache loaded models between runs
- Use safetensors format for faster I/O

## 📞 Getting Help

1. **Check Documentation**: Read the detailed analysis files
2. **Search Issues**: Look for similar problems in GitHub issues
3. **Create Issue**: Report bugs or ask questions
4. **Discussions**: Join mathematical discussions

## 🔄 Updating the Analysis

To update to the latest version:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🎓 Educational Resources

- **STARCODER2_ARCHITECTURE_ANALYSIS.md**: Complete architecture breakdown
- **LORA_VS_TASK_VECTOR_ANALYSIS.md**: Mathematical relationship explanation
- **EXPERIMENTAL_VALIDATION_SUMMARY.md**: Validation methodology and results

Happy analyzing! 🔬
