# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold: `pyproject.toml`, `.gitignore`, Python 3.10 pin
- 7-category multi-label schema and conversation data model
- Role-aware tokenizer with four special tokens (`[USER]`, `[CHAR]`, `[SYSTEM]`, `[CURRENT]`)
- Jina v2 base encoder wrapper with current-message pooling
- Focal loss with per-class positive weights
- Training loop with layer-wise learning-rate decay and fp16 mixed precision
- Dataset loaders for Jigsaw, Civil Comments, Measuring-Hate-Speech, Salad-Data, ProsocialDialog, and Anthropic HH-RLHF
- Dataset aggregator with deduplication, split, and stats reporting
- Trajectory augmentation pipeline (benign-context padding and edgy-context hard negatives)
- Synthetic trajectory generation with Anthropic, Ollama, and mock backends, and a human-review gate
- ONNX export with INT8 dynamic quantization and parity checks
- Python inference API (`sage.inference.Sage`)
- Rust inference crate (`rust/sage-rs`) with ort-based session and role-aware tokenizer
- Documentation: architecture specification, data-license attribution
