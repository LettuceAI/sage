# Contributing

Thanks for your interest in `sage`. This document describes how to get set up, run the test suite, and submit changes.

## Setup

```bash
pyenv local 3.10.13   # or any 3.10+
python -m venv venv
source venv/bin/activate
pip install -e '.[train,export,dev]'
pre-commit install
```

The Rust crate lives under `rust/sage-rs`. It is checked independently:

```bash
cd rust/sage-rs
cargo check
cargo test
```

## Development loop

```bash
make test     # pytest
make lint     # ruff check + mypy
make format   # ruff format
make rust     # cargo check + cargo test
```

Run the full suite (`make check`) before opening a pull request.

## Pull requests

- One logical change per pull request.
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- All new behavior should be covered by a unit test.
- Public APIs (anything exported from `sage.__init__`, `training.data.__init__`, or the Rust crate root) must carry a docstring.

## Reporting bugs

Use the "Bug report" issue template. Please include:

- The input that reproduces the issue (or a minimized version of it)
- The expected behavior
- The actual behavior, including any error output

Security-sensitive issues should follow [`SECURITY.md`](SECURITY.md) instead of the public issue tracker.

## Requesting features

Use the "Feature request" issue template. Before investing time in implementation, please open an issue describing the motivation and proposed scope so we can discuss it first.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
