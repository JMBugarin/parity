# M0: pinned versions and endpoint verification

Checked 2026-11-20 against the pinned releases, both from source and against a running vLLM.

## Pins

| Component | Version | Commit or digest |
|---|---|---|
| vLLM | v0.34.0 (2026-11-16) | `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| vLLM CPU image (verification) | `vllm/vllm-openai-cpu:v0.34.0`, linux/arm64 | `sha256:1640fdc27ae499572f611ffd4dc55e0b07ce97608f33566d5834c98671b115d4` |
| tokenizers in that image | 0.23.3 | — |
| llm-d-inference-sim | v1.2.0 (2026-11-10) | `6cc170fcec957148d3adfc7fc9125428b75d2899` |
| Go | 1.27.4 | — |
| Model (verification) | Qwen/Qwen3-0.6B, bf16 on CPU | same tokenizer as Qwen3-1.7B |

Command: `docker run --shm-size=2g -p 8100:8000 vllm/vllm-openai-cpu:v0.34.0 Qwen/Qwen3-0.6B --enable-scale-out --max-model-len 4096 --dtype bfloat16 --served-model-name qwen3-0.6b`

## vLLM v0.34.0

| Needed by the gateway | Source | Live check |
|---|---|---|
| `POST /v1/chat/completions/render` returns a `GenerateRequest` (token IDs, resolved `sampling_params`, model) | `entrypoints/scale_out/render/api_router.py` | yes, `raw/` and `internal/upstream/testdata/render.json` |
| `POST /inference/v1/generate`, registered by `--enable-scale-out` | `entrypoints/common/cli_args.py` (`enable_scale_out`), `scale_out/token_in_token_out/api_router.py` | yes |
| Streaming chunks carry `token_ids` and, with `output_mode: "text"`, `text` in the same choice | `serving.py` `serve_tokens_stream_generator` | yes, `internal/sse/testdata/vllm-generate-text.sse` |
| `output_mode: "tokens"` returns IDs only | same | yes, `vllm-generate-tokens.sse` |
| Stream ends with a finish chunk, a usage chunk (`stream_options.include_usage`) and `data: [DONE]`; errors come as `data: {"error": …}` | same | yes |
| `POST /inference/v1/abort_requests {"request_ids": [...]}` | `api_router.py` | yes: the stream ends with `finish_reason: "abort"`. The ID must be the engine's, `generate-tokens-<id>`; the bare ID is ignored. |
| `POST /detokenize` | `serve/tokenize/api_router.py` | yes. It does **not** skip special tokens (`[19641,151645]` → `"Three<|im_end|>"`). |
| Request ID can come from `X-Request-Id` | `serve/engine/serving.py` `_base_request_id` | — |
| `sampling_params.output_kind` is set from `stream` by the generate handler, and `stop` strings make the engine hold back text (`output_text_buffer_length`) | `serving.py`, render output | yes (render returns `output_kind: 2`, `output_text_buffer_length: 1` with a stop string) |

## llm-d-inference-sim v1.2.0

Serves `POST /inference/v1/generate` and `POST /v1/chat/completions/render` (`pkg/engine/vllm/transport.go`). Differences that matter here:

- No `output_mode`: chunks carry `token_ids` only, so the gateway runs it with `--output-mode tokens` (placeholder text).
- No `/inference/v1/abort_requests` and no `/detokenize`; aborting is by closing the connection.
- Tokens come from a dataset, not from the prefix, so it cannot continue a sequence. It is used for latency and metrics realism, not as the continuity oracle.

## Finding: resuming inside a multi-byte character

Qwen3 splits " 🦊" into three tokens (11162, 99, 232). In `output_mode: "text"`, vLLM streams `""` for the first two and `" 🦊"` with the third (`raw/utf8-uninterrupted.sse`).

Resuming generation with the prompt plus the output so far (the splice's request) behaves differently depending on where the output ends (`raw/utf8-resume-k*.sse`):

| Output tokens included in the resume prompt | First text chunk from the resumed request |
|---|---|
| 11, ending before 11162 | `""`, `""`, `" 🦊"`: identical to the uninterrupted run |
| 12, ending with 11162 | the **entire prompt text** (`user\nReply with only these characters…`) followed by the output so far and `🦊` |
| 13, ending with 11162, 99 | the same: the whole prompt again |

The cause is in v0.34.0's `FastIncrementalDetokenizer`: it primes `tokenizers.decoders.DecodeStream(ids=prompt_token_ids)` in one step. When the prompt's decoded text ends in U+FFFD (an incomplete character), that step emits nothing and never sets the stream's prefix, so the first complete emission returns the decode of the whole prompt. A naive token-level splice therefore leaks the prompt into the client's stream whenever it lands inside a character.

Consequence for the design: the gateway commits a token only once the engine has released its text, and a resume prompt contains committed tokens only. Splices never land inside a character, and the case is covered by detsim, which reproduces this detokenizer (`internal/detsim/detok.go`, `TestDecodeStreamReproducesVLLM`).
