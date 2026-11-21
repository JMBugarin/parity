# Parity

[![ci](https://github.com/JMBugarin/parity/actions/workflows/ci.yml/badge.svg)](https://github.com/JMBugarin/parity/actions/workflows/ci.yml)

A Go gateway for vLLM that keeps a streaming response going when the replica serving it dies. It resends the prompt plus the tokens already sent to a healthy replica and splices the continuation into the same client stream.

Status: M0, pinned and verified. See [docs/guarantees.md](docs/guarantees.md) for what it promises and [docs/evidence/M0](docs/evidence/M0/VERIFICATION.md) for the endpoint checks.

## Pinned versions

| Component | Version |
|---|---|
| vLLM | v0.34.0 (`--enable-scale-out`: `/v1/chat/completions/render`, `/inference/v1/generate`, `/inference/v1/abort_requests`) |
| llm-d-inference-sim | v1.2.0 |
| Go | 1.27.4 |

## License

MIT
