# Security policy

## Supported versions

Pre-1.0. Only `main` receives fixes.

## Reporting

Use GitHub's private vulnerability reporting: <https://github.com/MegalithOfficial/sage/security/advisories/new>.

Include: description, impact, reproduction steps, suggested mitigation.

Acknowledgement within three business days. We coordinate disclosure before public discussion.

## Scope

In scope:
- Python package, Rust crate, training pipeline
- Adversarial inputs that reliably bypass classification

Out of scope:
- Compromised-host scenarios
- Rate limiting / denial of service at the hosting layer (we don't ship a service)
- Upstream dependency issues (report to the respective project)
