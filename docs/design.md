# Design (stub)

Parity sits between OpenAI-compatible clients and vLLM replicas. It renders the chat template once, then drives generation through vLLM's token-in/token-out API, so that a stream can continue on another replica from the exact tokens already sent.

```
client ─SSE─▶ /v1/chat/completions ─▶ render once (chat template → prompt IDs)
                 │
                 ▼
   attempt k: POST /inference/v1/generate {token_ids, sampling_params, stream, output_mode:"text"} ──▶ replica
                 │  chunks: {token_ids[], text}
                 ▼
   splice state: committed IDs · sent text · held-back text · max_tokens budget
                 │  failure? (reset / EOF without finish / 5xx / stall / drain)
                 ▼
   abort old → pick another replica → generate {prompt ++ committed, max_tokens − len(committed)} → continue
```

To be expanded with the M3 splice engine: request lifecycle, the attempt state machine, failure detection, the commit rule, text reconciliation and stop strings.
