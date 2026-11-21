# Guarantees (v0)

What Parity promises a client, under which conditions, and what it does not promise. v0 is the target from the spec; each row gains an evidence link as its experiment lands.

| Property | Promise | Conditions | Evidence |
|---|---|---|---|
| Continuity | One SSE stream per request, with no lost, duplicated or reordered tokens. The text sent equals the detokenized token IDs sent. | `n = 1`, at most 2 failovers (`--max-splices`), within the request deadline | E1 |
| Identity | The resumed stream is byte-identical to the uninterrupted one. | Greedy decoding, no frequency or presence penalties, the same image, flags and attention backend on every replica, batch invariance on. Measured (E3), not assumed. | E3 |
| Sampled decoding | Continuity only. The continuation is a valid sample, not the same one, because a seeded RNG restarts at the splice. | `temperature > 0` | E1 |
| Stop strings | Enforced at the gateway, so a stop string that straddles a splice still stops the stream. | Always. Engines get `stop_token_ids` only. | E1 |
| Usage | `prompt_tokens` is the original prompt and `completion_tokens` counts every token sent. Re-prefilled tokens are reported separately (`usage.parity.reprefill_tokens`). | Always | E1 |
| Cancellation | A client disconnect aborts the upstream request. A replica abandoned after a stall is aborted before the stream resumes elsewhere. | Best effort if that replica is unreachable | E1 |

## Not promised

- The client's own connection dropping. There is no `Last-Event-ID` resume.
- A gateway crash. Streams on a gateway that dies are lost; a gateway that is drained finishes them (`--shutdown-grace`).
- `n > 1`, beam search, logprobs. The gateway rejects these with 400.
- Tool-call parsing across a splice. The gateway returns the model's raw text.
- Moving KV cache between replicas. Parity re-prefills; E2 measures what that costs.

## Definitions

- **Committed token**: a token whose text the engine has released. Only committed tokens are sent to the client, counted in usage, or included in a resume prompt.
- **Splice**: a new attempt on another replica whose prompt is the original prompt followed by the committed tokens, with `max_tokens` reduced by their number.
- **Retry**: a new attempt before any token is committed. It does not count against the splice budget.
