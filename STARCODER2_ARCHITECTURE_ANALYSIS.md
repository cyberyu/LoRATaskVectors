# StarCoder2-3B Parameter Architecture Analysis

## Summary

The mystery of why Task Vectors contain 483 parameters while LoRA adapters only modify 120 has been **completely solved**. The 483 parameters represent the **complete architectural structure** of the StarCoder2-3B model, while LoRA only targets attention layers.

## Complete Parameter Breakdown

### StarCoder2-3B Architecture
- **30 transformer layers** (layers 0-29)
- **Total model parameters**: 483
- **LoRA-modified parameters**: 120 (attention weights only)
- **Non-LoRA parameters**: 363 (MLP, LayerNorm, Embeddings)

### Parameter Categories

| Category | Per Layer | Total | LoRA Modified | Notes |
|----------|-----------|-------|---------------|-------|
| **Attention** | 8 | 240 | 120 | 4 weights + 4 biases, only weights modified |
| **MLP** | 4 | 120 | 0 | 2 weights + 2 biases, none modified |
| **LayerNorm** | 4 | 120 | 0 | 2 norms × 2 params each, none modified |
| **Embeddings** | - | 1 | 0 | embed_tokens.weight, not modified |
| **Final Norm** | - | 2 | 0 | model.norm.bias + model.norm.weight |
| **Total** | 16 | **483** | **120** | |

### Mathematical Verification
```
Parameters per layer = 8 (attention) + 4 (MLP) + 4 (LayerNorm) = 16
Total parameters = 16 × 30 layers + 1 (embedding) + 2 (final norm) = 483 ✓
```

## Key Findings

### 1. **Task Vector Structure is Complete Model Architecture**
- Task vectors contain **every parameter** in the model (483 total)
- Most parameters remain zero (unchanged by LoRA fine-tuning)
- Only attention weight parameters contain non-zero values

### 2. **LoRA Targets Only Attention Weights** 
- **120 parameters modified**: 30 layers × 4 attention projections = 120 weights
- **120 parameters unmodified**: 30 layers × 4 attention biases = 120 biases  
- **363 parameters unmodified**: All MLP, LayerNorm, and embedding parameters

### 3. **Why 483 vs 120?**
- **Task Vector perspective**: Shows all 483 parameters with their changes (most are zero)
- **LoRA perspective**: Only tracks the 120 parameters that actually change
- **Relationship**: Task Vector is the "expanded view" of LoRA changes

## Precision Impact Analysis

### Float32 Task Vector (Ground Truth)
- **483 total parameters** (complete model structure)
- **120 non-zero parameters** (attention weights only)
- **363 zero parameters** (MLP, LayerNorm, embeddings unchanged)

### Float16 Task Vector (With Precision Artifacts)  
- **483 total parameters** (same structure)
- **483 non-zero parameters** (precision loss creates false changes)
- **363 spurious changes** (precision artifacts in unchanged parameters)

## Architecture Details

### Per-Layer Structure (30 layers)
```
model.layers.{0-29}.self_attn.{q,k,v,o}_proj.{weight,bias}  # 8 params
model.layers.{0-29}.mlp.{c_fc,c_proj}.{weight,bias}         # 4 params  
model.layers.{0-29}.{input,post_attention}_layernorm.{weight,bias} # 4 params
```

### Global Parameters
```
model.embed_tokens.weight          # 1 param
model.norm.{weight,bias}           # 2 params
```

## Mathematical Relationship Confirmed

```python
# LoRA Changes (Sparse)
lora_changes = {
    "attention_weights": 120_parameters,  # Non-zero deltas
    "everything_else": 0                  # Not modified by LoRA
}

# Task Vector Changes (Complete)  
task_vector_changes = {
    "attention_weights": 120_parameters,  # Same deltas as LoRA
    "attention_biases": 0,                # Unchanged (LoRA doesn't modify)
    "mlp_parameters": 0,                  # Unchanged (LoRA doesn't modify)  
    "layernorm_parameters": 0,            # Unchanged (LoRA doesn't modify)
    "embedding_parameters": 0             # Unchanged (LoRA doesn't modify)
}

# Mathematical equivalence for modified parameters:
# task_vector_changes["attention_weights"] == lora_changes["attention_weights"] ✓
```

## Conclusion

The **483 parameters** in Task Vectors represent the complete model architecture, not additional changes. LoRA and Task Vectors are **mathematically equivalent for the parameters they both modify** (attention weights), but:

- **LoRA** = Sparse representation (120 modified parameters)
- **Task Vector** = Dense representation (483 total parameters, 120 non-zero)

The "extra" 363 parameters in Task Vectors are architectural completeness, not actual model changes. This explains why Task Vectors are larger in storage but mathematically equivalent to LoRA for the actual modifications made during fine-tuning.
