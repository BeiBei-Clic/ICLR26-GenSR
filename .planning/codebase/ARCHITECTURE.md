# Architecture

**Analysis Date:** 2025-06-09

## Pattern Overview

**Overall:** Hybrid Encoder-Decoder VAE with Evolutionary Optimization

**Key Characteristics:**
- Latent space-based equation generation using Conditional Variational Autoencoder (CVAE)
- Dual-branch architecture for numerical data and symbolic equations
- Evolution Strategy (CMA-ES) optimization over latent space
- BFGS refinement for constant optimization
- Multi-stage pipeline: training → latent encoding → evolutionary search → refinement

## Layers

**Data Encoding Layer:**
- Purpose: Encode numerical (X, Y) data points into fixed-length representations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/embedders.py`
- Contains: `NumericalEmbedder` class with float encoding logic
- Depends on: Environment configuration, float encoder
- Used by: CVAE encoder during training and inference

**Equation Encoding Layer:**
- Purpose: Encode symbolic equations (trees) into token embeddings
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/embedders.py`
- Contains: Token embedding layer, positional embeddings
- Depends on: Equation vocabulary from environment
- Used by: CVAE encoder during training

**CVAE Encoder:**
- Purpose: Learn latent representations of (data, equation) pairs
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/cvae.py`
- Contains: `CVAEDE_SR` class with Transformer encoder and latent projection
- Depends on: Data encoder, equation encoder, Transformer layers
- Used by: Trainer for VAE training, inference pipeline for encoding

**Feature Fusion Layer:**
- Purpose: Fuse latent (μ, logvar) into decoder-ready representation
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/feature_fusion.py`
- Contains: `FeatureFusion` class
- Depends on: Latent dimension, decoder embedding dimension
- Used by: Decoder for equation generation

**Decoder Layer:**
- Purpose: Generate equation tokens from latent representation
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/transformer.py`
- Contains: `TransformerModel_VAE` decoder with cross-attention
- Depends on: Fused latent representation, token embeddings
- Used by: Training loss computation, inference generation

**Optimization Layer:**
- Purpose: Search latent space for optimal equations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py`
- Contains: CMA-ES implementation, population initialization, candidate evaluation
- Depends on: Pretrained CVAE model, environment for equation conversion
- Used by: Inference pipeline for final equation discovery

**Refinement Layer:**
- Purpose: Optimize constants in discovered equations
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/const_opt.py`
- Contains: BFGS-based constant refinement logic
- Depends on: Environment for equation-to-function conversion
- Used by: LSO optimization pipeline

## Data Flow

**Training Flow:**

1. **Data Sampling:** `Trainer.get_batch()` samples (X, Y, equation) tuples from environment
2. **Encoding:** Data encoder processes X,Y pairs; equation encoder tokenizes equations
3. **CVAE Forward Pass:**
   - Encoder produces prior (from data only) and posterior (from data + equation)
   - KL divergence computed between prior and posterior
4. **Decoding:** Decoder reconstructs equations from both prior and posterior latents
5. **Loss Computation:** Combined loss = prediction_loss(prior) + prediction_loss(posterior) + kl_loss
6. **Optimization:** Adam optimizer with gradient clipping updates CVAE parameters

**Inference Flow:**

1. **Data Encoding:** Input (X, Y) data → CVAE prior encoder → latent (μ, logvar)
2. **Population Initialization:**
   - Sample-based: Encode subsampled data points
   - Noise-based: Add noise to Y data or latent space
3. **Evolutionary Search:**
   - CMA-ES generates candidate latents
   - Each latent decoded to equation via decoder
   - Equations evaluated for fitness (R² - complexity_penalty)
   - Best candidates guide next generation
4. **Early Stopping:** Stop when R² threshold reached or max iterations exceeded
5. **Refinement:** BFGS optimizes constants in best-discovered equations
6. **Output:** Final equation with fitted constants and metrics

**State Management:**
- Training: VAE parameters, optimizer state, learning rate schedule
- Inference: Best equation skeleton, constants, fitness tracking, skeleton deduplication

## Key Abstractions

**Latent Space:**
- Purpose: Continuous representation space where each point maps to an equation
- Examples: `prior_mu`, `post_mu` tensors of shape (batch, latent_dim)
- Pattern: Gaussian distribution with (μ, logvar) parameters

**Equation Skeleton:**
- Purpose: Template of equation structure without constant values
- Examples: `skeleton_candidate` in `LSO_fit.py:gen2eq()`
- Pattern: Sympy expression tree with placeholder constants

**CMA-ES Population:**
- Purpose: Set of candidate solutions evolving over iterations
- Examples: `pop` tensor in `lso_fit_es_covfromvae_fit()`
- Pattern: (N, latent_dim) tensor sampled from Gaussian distribution

**Beam Search:**
- Purpose: Generate multiple equation candidates from single latent
- Examples: `beam_size` parameter in decoder generation
- Pattern: Parallel decoding with top-K hypothesis tracking

## Entry Points

**Training Entry Point:**
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/train.py:main()`
- Triggers: Direct script execution
- Responsibilities: Initialize environment, modules, trainer, run training epochs, handle checkpointing

**Direct Evaluation Entry Point:**
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/direct_eval.py:__main__()`
- Triggers: Script execution with evaluation parameters
- Responsibilities: Load pretrained model, run PMLB evaluation, save results

**Batch Inference Entry Point:**
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/experiments/pmlb/pmlb_batch_inference.py`
- Triggers: Script execution for batch processing
- Responsibilities: Process multiple datasets, generate CSV results

**LSO Optimization Entry Point:**
- Location: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py:lso_fit_es_covfromvae_fit()`
- Triggers: Called during inference for each dataset
- Responsibilities: Execute evolutionary search, return best equation

## Error Handling

**Strategy:** Graceful degradation with fallback to default values

**Patterns:**
- **NaN Detection:** Training checks for NaN loss and logs warnings
- **Empty Results:** Inference handles failed candidates with placeholder zeros
- **Checkpoint Recovery:** Model loading handles missing keys with strict=False
- **Skeleton Deduplication:** Prevents redundant evaluation of same equation structures

**Exception Handling:**
- Try-except in equation evaluation with fallback to failed status
- Timeout handling for equation simplification
- Gradient clipping for training stability

## Cross-Cutting Concerns

**Logging:** Weights & Biases integration for experiment tracking
**Validation:** R²-based early stopping during inference
**Authentication:** None (local computation)
**Distributed Training:** PyTorch DDP with gradient accumulation support
**Checkpointing:** Periodic model saves with optimizer state

---

*Architecture analysis: 2025-06-09*
