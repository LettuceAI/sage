# Security policy

## Supported versions

`sage` is pre-1.0 software. Only the `main` branch receives security fixes.

## Reporting a vulnerability

Please report security issues privately, not through the public issue tracker.

- **Email:** security@lettuce.ai
- **GitHub private vulnerability report:** <https://github.com/LettuceAI/sage/security/advisories/new>

Include:

- A description of the issue and its potential impact
- Steps to reproduce, including model inputs or training data patterns where relevant
- Any suggested mitigation

You should receive an acknowledgement within three business days. We will work with you on a disclosure timeline and coordinate a fix before public disclosure.

## Scope

In scope:

- The `sage` Python package and the `sage` Rust crate
- The training and export pipelines under `training/`
- Vulnerabilities in exported model artifacts (e.g. adversarial inputs that reliably bypass classification for a protected category)

Out of scope:

- Findings that rely on a compromised machine running `sage` — e.g. a malicious local ONNX file swapped in by an attacker with filesystem access
- Rate-limiting or denial-of-service at the hosting layer (this library does not provide a network service)
- Issues in upstream dependencies — please report those to the respective projects

## Model safety reports

Adversarial prompts, evasions, and persistent mislabels that affect user safety are welcome. Please use the vulnerability channel above so we can validate and reproduce the issue before any public discussion.
