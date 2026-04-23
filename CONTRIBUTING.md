# Contributing

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e '.[train,export,dev]'
pre-commit install
```

Rust crate:

```bash
cd rust/sage-rs && cargo check && cargo test
```

## Before opening a PR

```bash
make check
```

Runs ruff, mypy, pytest, and `cargo check`.

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- One logical change per PR.
- New behavior needs a unit test.
- Public API changes require a docstring.

## Issues

Use the bug report or feature request template. For security issues, see [SECURITY.md](SECURITY.md).
