# Codebase Structure

**Analysis Date:** 2025-06-09

## Directory Layout

```
/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/
├── symbolicregression/          # Core library package
│   ├── model/                   # Model architectures
│   ├── envs/                    # Environment and data generation
│   ├── trainer_vae.py          # Training logic
│   ├── optim.py                # Optimizer configurations
│   ├── utils.py                # Utility functions
│   ├── logger.py               # Logging setup
│   ├── slurm.py                # SLURM cluster support
│   └── metrics.py              # Evaluation metrics
├── datasets/                    # Data storage
│   ├── pmlb/                   # PMLB benchmark datasets
│   └── feynman/                # Feynman equation datasets
├── experiments/                 # Experiment scripts
│   ├── pmlb/                   # PMLB batch inference
│   └── latent_space/           # Latent space analysis
├── scripts/                     # Execution scripts
├── Cola-DLM-main/              # External library (LLM-based SR)
├── train.py                    # Training entry point
├── direct_eval.py              # Direct evaluation entry point
├── LSO_fit.py                  # LSO optimization implementation
├── LSO_eval.py                 # LSO evaluation utilities
├── const_opt.py                # Constant refinement (BFGS)
├── model.py                    # High-level model wrapper
├── parsers.py                  # Command-line argument parsing
├── cma_es_modular.py           # CMA-ES implementation
├── weights/                    # Pretrained model checkpoints
├── dump/                       # Training output directory
├── eval_result/                # Evaluation results storage
└── wandb/                      # Weights & Biases local cache
```

## Directory Purposes

**symbolicregression/:**
- Purpose: Core library package containing all model and training logic
- Contains: Model architectures, training infrastructure, utilities
- Key files: `model/__init__.py`, `trainer_vae.py`, `envs/__init__.py`

**symbolicregression/model/:**
- Purpose: Neural network architectures for CVAE and Transformer components
- Contains: CVAE encoder, Transformer decoder, embedders, feature fusion
- Key files: `cvae.py`, `transformer.py`, `embedders.py`, `feature_fusion.py`

**symbolicregression/envs/:**
- Purpose: Equation generation environment and data processing
- Contains: Symbolic equation generation, tree manipulation, encoders/decoders
- Key files: `environment.py`, `generators.py`, `simplifiers.py`, `encoders.py`

**datasets/:**
- Purpose: Storage for benchmark datasets (Feynman, PMLB, Strogatz)
- Contains: Compressed dataset files (.tsv.gz), dataset metadata
- Key files: `pmlb/datasets/`, `feynman/`

**experiments/:**
- Purpose: Experiment scripts for batch processing and analysis
- Contains: Batch inference, result summarization, latent space visualization
- Key files: `pmlb/pmlb_batch_inference.py`, `latent_space/latent_distribution.py`

**scripts/:**
- Purpose: Shell scripts for common workflows
- Contains: Training, evaluation, and model downloading scripts
- Key files: `train.sh`, `eval.sh`, `bootstrap_pretrained.sh`

**Cola-DLM-main/:**
- Purpose: External library for LLM-based symbolic regression
- Contains: Separate package with its own structure
- Key files: `cola_dlm/`, `examples/`, `scripts/`

## Key File Locations

**Entry Points:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/train.py`: Main training script
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/direct_eval.py`: PMLB evaluation script
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/model.py`: High-level model wrapper

**Configuration:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/parsers.py`: Command-line argument parser
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/requirements.txt`: Python dependencies

**Core Logic:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/trainer_vae.py`: Training loop
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py`: Evolutionary optimization
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/const_opt.py`: Constant refinement

**Model Components:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/cvae.py`: CVAE architecture
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/transformer.py`: Transformer layers
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/embedders.py`: Data encoding

**Environment:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/envs/environment.py`: Data environment
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/envs/generators.py`: Equation generation
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/envs/simplifiers.py`: Equation simplification

**Testing:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/tests/`: Test files (minimal coverage)

**Documentation:**
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/README.md`: Project overview
- `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/RUN.md`: Running instructions

## Naming Conventions

**Files:**
- **Training/entry files:** `*.py` (e.g., `train.py`, `direct_eval.py`)
- **Model components:** `*.py` in `symbolicregression/model/` (e.g., `cvae.py`, `transformer.py`)
- **Utilities:** `*.py` with descriptive names (e.g., `utils.py`, `metrics.py`)
- **Experiments:** `*.py` in `experiments/` with descriptive names (e.g., `pmlb_batch_inference.py`)
- **Scripts:** `*.sh` in `scripts/` (e.g., `train.sh`, `eval.sh`)

**Directories:**
- **Library code:** Lowercase with underscores (e.g., `symbolicregression/`, `envs/`)
- **Experiments:** Descriptive names (e.g., `experiments/pmlb/`, `experiments/latent_space/`)
- **Data:** Descriptive names (e.g., `datasets/`, `weights/`, `dump/`)

**Classes:**
- **Model classes:** PascalCase (e.g., `CVAEDE_SR`, `TransformerModel_VAE`, `NumericalEmbedder`)
- **Trainer classes:** PascalCase (e.g., `Trainer`, `LoadParameters`)
- **Environment classes:** PascalCase (e.g., `FunctionEnvironment`)

**Functions:**
- **Core functions:** snake_case (e.g., `lso_fit_es_covfromvae_fit`, `build_modules`, `encode`)
- **Utility functions:** snake_case (e.g., `to_cuda`, `bool_flag`, `initialize_exp`)

**Variables:**
- **Parameters:** snake_case (e.g., `latent_dim`, `batch_size`, `learning_rate`)
- **Tensor variables:** Descriptive names (e.g., `prior_mu`, `post_logvar`, `encoded_y`)
- **Loop variables:** Short names (e.g., `i`, `idx`, `bs` for batch size)

## Where to Add New Code

**New Model Architecture:**
- Primary code: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/your_model.py`
- Tests: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/tests/test_your_model.py`
- Registration: Add to `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/__init__.py`

**New Optimization Strategy:**
- Implementation: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/your_optimizer.py`
- Integration: Update `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/LSO_fit.py`
- Registration: Add option to `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/parsers.py`

**New Dataset/Environment:**
- Environment: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/envs/your_env.py`
- Registration: Add to `ENVS` dict in `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/envs/__init__.py`

**New Experiment:**
- Implementation: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/experiments/your_experiment/your_script.py`
- Script: Create shell script in `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/scripts/your_script.sh`

**New Component/Module:**
- Implementation: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/your_component.py`
- Integration: Import in `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/__init__.py`
- Usage: Update `build_modules()` in `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/__init__.py`

**Utilities:**
- Shared helpers: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/utils.py`
- Model-specific utilities: `/home/xyh/Symbolic_Regression/E2E/ICLR26-GenSR/symbolicregression/model/utils_wrapper.py`

## Special Directories

**weights/:**
- Purpose: Stores pretrained model checkpoints
- Generated: No (manually downloaded or trained)
- Committed: No (in .gitignore)
- Contents: `checkpoint.pth` (main pretrained model)

**dump/:**
- Purpose: Training output directory (checkpoints, logs)
- Generated: Yes (created during training)
- Committed: No (in .gitignore)
- Contents: Experiment directories with periodic checkpoints

**eval_result/:**
- Purpose: Evaluation results storage
- Generated: Yes (created during evaluation)
- Committed: No (in .gitignore)
- Contents: CSV files with evaluation metrics

**wandb/:**
- Purpose: Weights & Biases local cache
- Generated: Yes (automatically created by wandb)
- Committed: No (in .gitignore)
- Contents: Offline run data and artifacts

**.planning/:**
- Purpose: Planning and analysis documents
- Generated: Yes (created by GSD tools)
- Committed: Yes (tracked in git)
- Contents: Codebase analysis documents (this file)

**Cola-DLM-main/:**
- Purpose: External LLM-based symbolic regression library
- Generated: No (copied from external source)
- Committed: Yes (as submodule or copied)
- Contents: Complete separate package structure

---

*Structure analysis: 2025-06-09*
