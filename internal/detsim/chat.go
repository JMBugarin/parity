package detsim

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// handleChat serves OpenAI chat completions directly, with stop strings
// applied in the engine. The gateway's passthrough mode uses it as the
// no-failover baseline.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	s.Stats.Requests.Add(1)
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, "replica is draining")
		return
	}
	var req chatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 16<<20)).Decode(&req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	prompt, sp, err := s.render(&req)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	plan, err := ParsePlan(r.Header.Get("X-Detsim-Fault"))
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	fault := plan.For(1)
	if fault.Kind == "5xx" {
		writeError(w, http.StatusServiceUnavailable, "injected 5xx")
		return
	}
	id := "chatcmpl-" + randomID()
	rn, err := s.register(id)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	s.Stats.Running.Add(1)
	defer func() {
		s.Stats.Running.Add(-1)
		s.unregister(rn)
	}()

	maxTok := sp["max_tokens"].(int)
	var sam Sampling
	sam.Temperature = sp["temperature"].(float64)
	if v, ok := sp["seed"].(int64); ok {
		sam.Seed, sam.HasSeed = uint64(v), true
	}
	sam.IgnoreEOS = req.IgnoreEOS
	stops := parseStops(req.Stop)
	holdLen := 0
	for _, st := range stops {
		holdLen = max(holdLen, len(st)-1)
	}
	created := time.Now().Unix()
	model := s.cfg.ModelName
	ctx := r.Context()
	flusher, _ := w.(http.Flusher)

	var out strings.Builder
	var outIDs []int
	sentText := 0
	finish := "length"
	var stopReason any
	ds := NewDecodeStream(s.Vocab, prompt, true, s.cfg.FixedDetok)
	st := s.Model.Prefix(prompt)

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.WriteHeader(http.StatusOK)
	}
	writeChunk := func(delta map[string]any, ids []int, fin *string, extra map[string]any) bool {
		choice := map[string]any{"index": 0, "delta": delta, "logprobs": nil, "finish_reason": fin}
		if req.ReturnTokenIDs && ids != nil {
			choice["token_ids"] = ids
		}
		m := map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": []any{choice}}
		for k, v := range extra {
			m[k] = v
		}
		b, _ := json.Marshal(m)
		if _, err := w.Write(append(append([]byte("data: "), b...), "\n\n"...)); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	if req.Stream {
		extra := map[string]any{}
		if req.ReturnTokenIDs {
			extra["prompt_token_ids"] = prompt
		}
		if !writeChunk(map[string]any{"role": "assistant", "content": ""}, nil, nil, extra) {
			return
		}
	}
	prefill := s.cfg.TTFTBase + time.Duration(len(prompt)-s.cache.Match(prompt))*s.cfg.PrefillPerToken
	if _, gone := sleep(ctx, rn, prefill); gone {
		return
	}
	s.cache.Insert(prompt)
	n := 0
	for i := 0; i < maxTok; i++ {
		if fault.Kind == "die" && i == fault.At {
			reset(w)
			return
		}
		if i > 0 {
			if _, gone := sleep(ctx, rn, s.itl(&genParams{sampling: sam, fault: fault}, i)); gone {
				return
			}
		}
		tok := s.Model.Next(st, i, sam)
		st = st.Push(tok)
		n++
		outIDs = append(outIDs, tok)
		var fin *string
		if tok == TokIMEnd {
			f := "stop"
			fin, finish = &f, f
		} else if i == maxTok-1 {
			f := "length"
			fin, finish = &f, f
		}
		out.WriteString(ds.Step(tok))
		text := out.String()
		cut := -1
		for _, stop := range stops {
			if k := strings.Index(text[max(0, sentText-holdLen):], stop); k >= 0 {
				pos := max(0, sentText-holdLen) + k
				if cut < 0 || pos < cut {
					cut = pos
					stopReason = stop
					if req.IncludeStopStr {
						cut = pos + len(stop)
					}
				}
			}
		}
		release := len(text) - holdLen
		if cut >= 0 {
			release = cut
			f := "stop"
			fin, finish = &f, f
		} else if fin != nil {
			release = len(text)
		}
		release = max(release, sentText)
		for release > sentText && release < len(text) && !utf8Start(text[release]) {
			release--
		}
		delta := text[sentText:release]
		sentText = release
		if req.Stream && (delta != "" || fin != nil) {
			d := map[string]any{}
			if delta != "" {
				d["content"] = delta
			}
			if !writeChunk(d, []int{tok}, nil, nil) {
				return
			}
		}
		if fin != nil {
			break
		}
	}
	usage := map[string]any{"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}
	content := out.String()[:sentText]
	if req.Stream {
		f := finish
		final := map[string]any{"index": 0, "delta": map[string]any{}, "logprobs": nil, "finish_reason": f, "stop_reason": stopReason}
		b, _ := json.Marshal(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": []any{final}})
		buf := append(append([]byte("data: "), b...), "\n\n"...)
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			b, _ = json.Marshal(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": []any{}, "usage": usage})
			buf = append(append(append(buf, "data: "...), b...), "\n\n"...)
		}
		buf = append(buf, "data: [DONE]\n\n"...)
		_, _ = w.Write(buf)
		return
	}
	resp := map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": content},
			"logprobs": nil, "finish_reason": finish, "stop_reason": stopReason, "token_ids": outIDs}},
		"usage": usage,
	}
	if req.ReturnTokenIDs {
		resp["prompt_token_ids"] = prompt
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(mustMarshal(resp))))
	_, _ = w.Write(mustMarshal(resp))
}

func utf8Start(b byte) bool { return b&0xC0 != 0x80 }

func mustMarshal(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}
