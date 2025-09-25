# HS-MAD: Hierarchical Symbolic-Conditioned Music Audio Diffusion

This repository implements the HS-MAD architecture described in the accompanying technical and experimental specifications. The codebase provides a complete training and inference pipeline for hierarchical symbol-conditioned latent diffusion models operating in the Descript Audio Codec latent space. Key components include the Hierarchical Symbolic Encoder (HSE), multi-resolution conditioning with skip-connection modulation (SCM), decoupled classifier-free guidance, and comprehensive evaluation tooling.

> 所有模块严格遵循《技术文档 v4.md》与《实验文档 v2.md》的架构、维度和训练细节要求。

## Repository layout

```
hs_mad/
├── configs/          # YAML configs for data, model, training, inference, evaluation
├── scripts/          # Dataset preparation, feature caching, benchmarking
├── src/hs_mad/       # Library source code
├── tests/            # Unit and integration tests
├── requirements.txt  # Python dependencies
├── pyproject.toml    # Build system and tooling configuration
└── README.md
```

## Getting started

1. Install dependencies with `pip install -r requirements.txt` (PyTorch 2.3+ required).
2. Configure dataset paths in `configs/data.yaml`.
3. Pre-cache symbolic features using `scripts/cache_features.py` if desired.
4. Launch training via `python -m hs_mad.train.main --config configs/train.yaml`.

Detailed usage instructions, experiment setups, and evaluation pipelines are documented inline within each module and configuration file.
