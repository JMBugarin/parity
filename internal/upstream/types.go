// Package upstream is the gateway's client for vLLM's token-in/token-out
// API: /v1/chat/completions/render, /inference/v1/generate,
// /inference/v1/abort_requests and /detokenize.
package upstream

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Rendered is the GenerateRequest that /v1/chat/completions/render returns.
// Fields the gateway does not own are kept as raw JSON and passed through.
type Rendered struct {
	Raw      map[string]json.RawMessage
	TokenIDs []int
	Sampling map[string]json.RawMessage
	Model    string
}

func ParseRendered(b []byte) (*Rendered, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("render response: %w", err)
	}
	r := &Rendered{Raw: raw}
	if err := json.Unmarshal(raw["token_ids"], &r.TokenIDs); err != nil || len(r.TokenIDs) == 0 {
		return nil, errors.New("render response: missing token_ids")
	}
	if sp, ok := raw["sampling_params"]; ok && string(sp) != "null" {
		if err := json.Unmarshal(sp, &r.Sampling); err != nil {
			return nil, fmt.Errorf("render response: sampling_params: %w", err)
		}
	}
	if r.Sampling == nil {
		r.Sampling = map[string]json.RawMessage{}
	}
	if m, ok := raw["model"]; ok {
		_ = json.Unmarshal(m, &r.Model)
	}
	return r, nil
}

// Int reads an integer sampling parameter.
func (r *Rendered) Int(key string) (int, bool) {
	v, ok := r.Sampling[key]
	if !ok || string(v) == "null" {
		return 0, false
	}
	var n int
	if err := json.Unmarshal(v, &n); err != nil {
		return 0, false
	}
	return n, true
}

func (r *Rendered) Float(key string) (float64, bool) {
	v, ok := r.Sampling[key]
	if !ok || string(v) == "null" {
		return 0, false
	}
	var f float64
	if err := json.Unmarshal(v, &f); err != nil {
		return 0, false
	}
	return f, true
}

func (r *Rendered) Bool(key string) (bool, bool) {
	v, ok := r.Sampling[key]
	if !ok || string(v) == "null" {
		return false, false
	}
	var b bool
	if err := json.Unmarshal(v, &b); err != nil {
		return false, false
	}
	return b, true
}

// Stops reads sampling_params.stop, which vLLM serializes as a string or a
// list of strings.
func (r *Rendered) Stops() []string {
	v, ok := r.Sampling["stop"]
	if !ok || string(v) == "null" {
		return nil
	}
	var one string
	if json.Unmarshal(v, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(v, &many)
	out := many[:0]
	for _, s := range many {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// Attempt describes one dispatch of the request to a replica.
type Attempt struct {
	RequestID string
	TokenIDs  []int
	MaxTokens int
	MinTokens int
	// OutputMode is "text" or "tokens".
	OutputMode string
}

// GenerateBody builds the /inference/v1/generate body for one attempt. Stop
// strings are removed because the gateway enforces them itself, so a stop
// string that straddles a splice still stops the stream.
func (r *Rendered) GenerateBody(a Attempt) ([]byte, error) {
	sp := make(map[string]json.RawMessage, len(r.Sampling)+2)
	for k, v := range r.Sampling {
		sp[k] = v
	}
	delete(sp, "stop")
	delete(sp, "output_kind")
	delete(sp, "include_stop_str_in_output")
	sp["output_text_buffer_length"] = json.RawMessage("0")
	sp["max_tokens"] = mustJSON(a.MaxTokens)
	if _, ok := r.Sampling["min_tokens"]; ok || a.MinTokens > 0 {
		sp["min_tokens"] = mustJSON(a.MinTokens)
	}
	if a.OutputMode == "text" {
		sp["detokenize"] = json.RawMessage("true")
	}
	body := make(map[string]json.RawMessage, len(r.Raw)+4)
	for k, v := range r.Raw {
		body[k] = v
	}
	body["request_id"] = mustJSON(a.RequestID)
	body["token_ids"] = mustJSON(a.TokenIDs)
	body["sampling_params"] = mustJSON(sp)
	body["stream"] = json.RawMessage("true")
	body["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	body["output_mode"] = mustJSON(a.OutputMode)
	return json.Marshal(body)
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// Chunk is one SSE event from /inference/v1/generate.
type Chunk struct {
	RequestID string          `json:"request_id"`
	Choices   []Choice        `json:"choices"`
	Usage     *Usage          `json:"usage"`
	Error     json.RawMessage `json:"error"`
}

type Choice struct {
	Index        int             `json:"index"`
	Text         *string         `json:"text"`
	TokenIDs     []int           `json:"token_ids"`
	FinishReason *string         `json:"finish_reason"`
	StopReason   json.RawMessage `json:"stop_reason"`
}

type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
}

// ErrorMessage extracts a message from vLLM's error bodies, which come as
// {"error": {...}} or the older {"object": "error", "message": ...}.
func ErrorMessage(b []byte) string {
	var v struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(b, &v) != nil {
		return strings.TrimSpace(string(b))
	}
	if len(v.Error) > 0 {
		var e struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(v.Error, &e) == nil && e.Message != "" {
			return e.Message
		}
		var s string
		if json.Unmarshal(v.Error, &s) == nil {
			return s
		}
	}
	if v.Message != "" {
		return v.Message
	}
	if len(v.Detail) > 0 {
		return string(v.Detail)
	}
	return strings.TrimSpace(string(b))
}
