package detsim

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Config sets the model identity, the latency model and the defaults of one
// detsim replica.
type Config struct {
	Name      string
	ModelName string
	// Seed is the model identity: replicas with the same seed behave like
	// replicas of the same image and weights.
	Seed      uint64
	EOSPeriod uint64

	TTFTBase        time.Duration
	PrefillPerToken time.Duration
	ITL             time.Duration
	ITLJitter       float64
	ITLPerActive    float64
	CacheBlocks     int
	MaxModelLen     int

	// FixedDetok emulates a DecodeStream that handles prompts ending inside
	// a character; the default reproduces vLLM v0.34.0.
	FixedDetok   bool
	DefaultFault string
	Logger       *slog.Logger
}

// Server serves one replica.
type Server struct {
	cfg   Config
	Model *Model
	Vocab *Vocab
	cache *PrefixCache
	mux   *http.ServeMux

	mu      sync.Mutex
	running map[string]*run

	draining atomic.Bool
	Stats    Stats
}

// Stats are cumulative counters, also exported on /metrics.
type Stats struct {
	Requests     atomic.Int64
	Running      atomic.Int64
	Aborted      atomic.Int64
	Disconnected atomic.Int64
	Faults       atomic.Int64
	Tokens       atomic.Int64
	CachedTokens atomic.Int64
	PromptTokens atomic.Int64
}

type run struct {
	id      string
	aborted chan struct{}
	once    sync.Once
}

func (r *run) abort() { r.once.Do(func() { close(r.aborted) }) }

func New(cfg Config) *Server {
	if cfg.ModelName == "" {
		cfg.ModelName = "detsim"
	}
	if cfg.MaxModelLen == 0 {
		cfg.MaxModelLen = 32768
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	v := NewVocab()
	s := &Server{
		cfg:     cfg,
		Vocab:   v,
		Model:   &Model{Vocab: v, Seed: cfg.Seed, EOSPeriod: cfg.EOSPeriod},
		cache:   NewPrefixCache(cfg.CacheBlocks),
		running: map[string]*run{},
		mux:     http.NewServeMux(),
	}
	s.mux.HandleFunc("POST /v1/chat/completions/render", s.handleRender)
	s.mux.HandleFunc("POST /inference/v1/generate", s.handleGenerate)
	s.mux.HandleFunc("POST /inference/v1/abort_requests", s.handleAbort)
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	s.mux.HandleFunc("POST /detokenize", s.handleDetokenize)
	s.mux.HandleFunc("POST /tokenize", s.handleTokenize)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mux.HandleFunc("GET /metrics", s.handleMetrics)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// Drain makes /health fail and rejects new generations; running ones go on.
func (s *Server) Drain() { s.draining.Store(true) }

func (s *Server) Draining() bool { return s.draining.Load() }

// WaitIdle blocks until no generation is running or ctx ends.
func (s *Server) WaitIdle(ctx context.Context) error {
	t := time.NewTicker(10 * time.Millisecond)
	defer t.Stop()
	for s.Stats.Running.Load() > 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
	return nil
}

func randomID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": msg, "type": http.StatusText(code), "param": nil, "code": code,
	}})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if s.draining.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{
		map[string]any{"id": s.cfg.ModelName, "object": "model", "owned_by": "detsim", "max_model_len": s.cfg.MaxModelLen},
	}})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	st := &s.Stats
	fmt.Fprintf(w, "detsim_requests_total %d\n", st.Requests.Load())
	fmt.Fprintf(w, "detsim_requests_running %d\n", st.Running.Load())
	fmt.Fprintf(w, "detsim_aborted_total %d\n", st.Aborted.Load())
	fmt.Fprintf(w, "detsim_disconnected_total %d\n", st.Disconnected.Load())
	fmt.Fprintf(w, "detsim_faults_total %d\n", st.Faults.Load())
	fmt.Fprintf(w, "detsim_generation_tokens_total %d\n", st.Tokens.Load())
	fmt.Fprintf(w, "detsim_prompt_tokens_total %d\n", st.PromptTokens.Load())
	fmt.Fprintf(w, "detsim_prefix_cache_hit_tokens_total %d\n", st.CachedTokens.Load())
	d := 0
	if s.draining.Load() {
		d = 1
	}
	fmt.Fprintf(w, "detsim_draining %d\n", d)
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	ids := s.Vocab.Encode(req.Prompt)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"count": len(ids), "max_model_len": s.cfg.MaxModelLen, "tokens": ids})
}

func (s *Server) handleDetokenize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	for _, t := range req.Tokens {
		if t < 0 || t >= s.Vocab.Size() {
			writeError(w, 400, "token id out of range")
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	// Like vLLM's /detokenize, special tokens are not skipped.
	_ = json.NewEncoder(w).Encode(map[string]string{"prompt": s.Vocab.Decode(req.Tokens, false)})
}

func (s *Server) handleAbort(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RequestIDs json.RawMessage `json:"request_ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	var ids []string
	if json.Unmarshal(req.RequestIDs, &ids) != nil {
		var one string
		if err := json.Unmarshal(req.RequestIDs, &one); err != nil {
			writeError(w, 400, "request_ids must be a string or list")
			return
		}
		ids = []string{one}
	}
	s.mu.Lock()
	for _, id := range ids {
		if rn, ok := s.running[id]; ok {
			rn.abort()
		}
	}
	s.mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

// chatRequest is the subset of an OpenAI chat request detsim understands.
type chatRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	Seed                *int64          `json:"seed"`
	Stop                json.RawMessage `json:"stop"`
	StopTokenIDs        []int           `json:"stop_token_ids"`
	IgnoreEOS           bool            `json:"ignore_eos"`
	MinTokens           int             `json:"min_tokens"`
	IncludeStopStr      bool            `json:"include_stop_str_in_output"`
	N                   *int            `json:"n"`
	Stream              bool            `json:"stream"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	ReturnTokenIDs    bool  `json:"return_token_ids"`
	SkipSpecialTokens *bool `json:"skip_special_tokens"`
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

func (m chatMessage) text() string {
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	_ = json.Unmarshal(m.Content, &parts)
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func parseStops(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(raw, &many)
	return many
}

func (s *Server) render(req *chatRequest) (prompt []int, sp map[string]any, err error) {
	if len(req.Messages) == 0 {
		return nil, nil, errors.New("messages must not be empty")
	}
	if req.N != nil && *req.N != 1 {
		return nil, nil, errors.New("detsim supports n=1 only")
	}
	msgs := make([]Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = Message{Role: m.Role, Content: m.text()}
	}
	prompt = s.Vocab.ChatPrompt(msgs)
	if len(prompt) >= s.cfg.MaxModelLen {
		return nil, nil, fmt.Errorf("prompt has %d tokens; max_model_len is %d", len(prompt), s.cfg.MaxModelLen)
	}
	maxTok := s.cfg.MaxModelLen - len(prompt)
	if req.MaxCompletionTokens != nil {
		maxTok = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		maxTok = *req.MaxTokens
	}
	if maxTok < 1 {
		return nil, nil, errors.New("max_tokens must be at least 1")
	}
	if len(prompt)+maxTok > s.cfg.MaxModelLen {
		return nil, nil, fmt.Errorf("prompt (%d) + max_tokens (%d) exceeds max_model_len %d", len(prompt), maxTok, s.cfg.MaxModelLen)
	}
	temp := 1.0
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	sp = map[string]any{
		"temperature":    temp,
		"max_tokens":     maxTok,
		"stop":           nonNil(parseStops(req.Stop)),
		"stop_token_ids": nonNilInts(req.StopTokenIDs),
	}
	if req.TopP != nil {
		sp["top_p"] = *req.TopP
	}
	if req.Seed != nil {
		sp["seed"] = *req.Seed
	}
	if req.IgnoreEOS {
		sp["ignore_eos"] = true
	}
	if req.MinTokens > 0 {
		sp["min_tokens"] = req.MinTokens
	}
	if req.IncludeStopStr {
		sp["include_stop_str_in_output"] = true
	}
	if req.SkipSpecialTokens != nil && !*req.SkipSpecialTokens {
		sp["skip_special_tokens"] = false
	}
	return prompt, sp, nil
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func nonNilInts(s []int) []int {
	if s == nil {
		return []int{}
	}
	return s
}

func (s *Server) handleRender(w http.ResponseWriter, r *http.Request) {
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
	id := r.Header.Get("X-Request-Id")
	if id == "" {
		id = randomID()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"request_id":      "chatcmpl-" + id,
		"token_ids":       prompt,
		"sampling_params": sp,
		"model":           s.cfg.ModelName,
		"stream":          false,
		"output_mode":     "tokens",
	})
}

type genRequest struct {
	RequestID      string `json:"request_id"`
	TokenIDs       []int  `json:"token_ids"`
	SamplingParams struct {
		MaxTokens         *int            `json:"max_tokens"`
		MinTokens         int             `json:"min_tokens"`
		Temperature       *float64        `json:"temperature"`
		Seed              *int64          `json:"seed"`
		IgnoreEOS         bool            `json:"ignore_eos"`
		StopTokenIDs      []int           `json:"stop_token_ids"`
		SkipSpecialTokens *bool           `json:"skip_special_tokens"`
		Detokenize        *bool           `json:"detokenize"`
		Stop              json.RawMessage `json:"stop"`
	} `json:"sampling_params"`
	Stream        bool `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	OutputMode string `json:"output_mode"`
}

// genParams is a validated generation job.
type genParams struct {
	engineID    string
	prompt      []int
	max, min    int
	sampling    Sampling
	skipSpecial bool
	textMode    bool
	usage       bool
	fault       Fault
}

func (s *Server) parseGenerate(r *http.Request) (*genParams, int, error) {
	var req genRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&req); err != nil {
		return nil, 400, err
	}
	if len(req.TokenIDs) == 0 {
		return nil, 400, errors.New("token_ids must not be empty")
	}
	for _, t := range req.TokenIDs {
		if t < 0 || t >= s.Vocab.Size() {
			return nil, 400, fmt.Errorf("token id %d out of range", t)
		}
	}
	if len(parseStops(req.SamplingParams.Stop)) > 0 {
		return nil, 400, errors.New("detsim's generate endpoint does not implement stop strings; send stop_token_ids")
	}
	p := &genParams{prompt: req.TokenIDs, skipSpecial: true}
	base := r.Header.Get("X-Request-Id")
	if base == "" {
		base = req.RequestID
	}
	if base == "" {
		base = randomID()
	}
	p.engineID = "generate-tokens-" + base
	p.max = s.cfg.MaxModelLen - len(req.TokenIDs)
	if req.SamplingParams.MaxTokens != nil {
		p.max = *req.SamplingParams.MaxTokens
	}
	if p.max < 1 {
		return nil, 400, errors.New("max_tokens must be at least 1")
	}
	if len(req.TokenIDs)+p.max > s.cfg.MaxModelLen {
		return nil, 400, fmt.Errorf("prompt (%d) + max_tokens (%d) exceeds max_model_len %d", len(req.TokenIDs), p.max, s.cfg.MaxModelLen)
	}
	p.min = req.SamplingParams.MinTokens
	if t := req.SamplingParams.Temperature; t != nil {
		p.sampling.Temperature = *t
	} else {
		p.sampling.Temperature = 1
	}
	if sd := req.SamplingParams.Seed; sd != nil {
		p.sampling.Seed, p.sampling.HasSeed = uint64(*sd), true
	} else {
		var b [8]byte
		_, _ = rand.Read(b[:])
		for _, x := range b {
			p.sampling.Seed = p.sampling.Seed<<8 | uint64(x)
		}
	}
	p.sampling.IgnoreEOS = req.SamplingParams.IgnoreEOS
	if len(req.SamplingParams.StopTokenIDs) > 0 {
		p.sampling.StopTokens = map[int]bool{}
		for _, t := range req.SamplingParams.StopTokenIDs {
			p.sampling.StopTokens[t] = true
		}
	}
	if v := req.SamplingParams.SkipSpecialTokens; v != nil {
		p.skipSpecial = *v
	}
	switch req.OutputMode {
	case "", "tokens":
	case "text":
		if d := req.SamplingParams.Detokenize; d != nil && !*d {
			return nil, 400, errors.New("output_mode='text' requires sampling_params.detokenize to be true")
		}
		p.textMode = true
	default:
		return nil, 400, fmt.Errorf("output_mode %q is not supported by detsim", req.OutputMode)
	}
	if !req.Stream {
		return nil, 400, errors.New("detsim serves stream=true only")
	}
	p.usage = req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	spec := r.Header.Get("X-Detsim-Fault")
	if spec == "" {
		spec = s.cfg.DefaultFault
	}
	plan, err := ParsePlan(spec)
	if err != nil {
		return nil, 400, err
	}
	attempt, _ := strconv.Atoi(r.Header.Get("X-Parity-Attempt"))
	if attempt < 1 {
		attempt = 1
	}
	p.fault = plan.For(attempt)
	return p, 0, nil
}

func (s *Server) register(id string) (*run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.running[id]; dup {
		return nil, fmt.Errorf("request id %s is already running", id)
	}
	rn := &run{id: id, aborted: make(chan struct{})}
	s.running[id] = rn
	return rn, nil
}

func (s *Server) unregister(rn *run) {
	s.mu.Lock()
	if s.running[rn.id] == rn {
		delete(s.running, rn.id)
	}
	s.mu.Unlock()
}

// sleep waits for d, an abort, or the client going away.
func sleep(ctx context.Context, rn *run, d time.Duration) (aborted, gone bool) {
	if d <= 0 {
		select {
		case <-rn.aborted:
			return true, false
		case <-ctx.Done():
			return false, true
		default:
			return false, false
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return false, false
	case <-rn.aborted:
		return true, false
	case <-ctx.Done():
		return false, true
	}
}

func (s *Server) itl(p *genParams, i int) time.Duration {
	d := s.cfg.ITL
	if s.cfg.ITLPerActive > 0 {
		d = time.Duration(float64(d) * (1 + s.cfg.ITLPerActive*float64(s.Stats.Running.Load()-1)))
	}
	if s.cfg.ITLJitter > 0 && d > 0 {
		u := float64(splitmix(uint64(i)^p.sampling.Seed^0xA5A5)%10000)/10000*2 - 1
		d = time.Duration(float64(d) * (1 + s.cfg.ITLJitter*u))
	}
	if p.fault.Drip > 0 {
		d += p.fault.Drip
	}
	return d
}

type chunkJSON struct {
	RequestID  string       `json:"request_id"`
	Usage      *usageJSON   `json:"usage"`
	OutputMode string       `json:"output_mode"`
	Choices    []choiceJSON `json:"choices"`
}

type choiceJSON struct {
	Index        int             `json:"index"`
	FinishReason *string         `json:"finish_reason"`
	StopReason   json.RawMessage `json:"stop_reason"`
	TokenIDs     []int           `json:"token_ids"`
	Text         *string         `json:"text,omitempty"`
	Logprobs     json.RawMessage `json:"logprobs"`
}

type usageJSON struct {
	PromptTokens     int `json:"prompt_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

var jsonNull = json.RawMessage("null")

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	s.Stats.Requests.Add(1)
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, "replica is draining")
		return
	}
	p, code, err := s.parseGenerate(r)
	if err != nil {
		writeError(w, code, err.Error())
		return
	}
	if p.fault.Kind == "5xx" {
		s.Stats.Faults.Add(1)
		writeError(w, http.StatusServiceUnavailable, "injected 5xx")
		return
	}
	rn, err := s.register(p.engineID)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	s.Stats.Running.Add(1)
	defer func() {
		s.Stats.Running.Add(-1)
		s.unregister(rn)
	}()
	s.generate(w, r, p, rn)
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request, p *genParams, rn *run) {
	ctx := r.Context()
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if flusher != nil {
		flusher.Flush()
	}

	cached := s.cache.Match(p.prompt)
	s.Stats.CachedTokens.Add(int64(cached))
	s.Stats.PromptTokens.Add(int64(len(p.prompt)))
	prefill := s.cfg.TTFTBase + time.Duration(len(p.prompt)-cached)*s.cfg.PrefillPerToken + p.fault.SlowFirst
	if aborted, gone := sleep(ctx, rn, prefill); gone {
		s.Stats.Disconnected.Add(1)
		return
	} else if aborted {
		s.finishAborted(w, flusher, p, 0)
		return
	}
	s.cache.Insert(p.prompt)

	var ds *DecodeStream
	if p.textMode {
		ds = NewDecodeStream(s.Vocab, p.prompt, p.skipSpecial, s.cfg.FixedDetok)
	}
	st := s.Model.Prefix(p.prompt)
	seq := append([]int(nil), p.prompt...)
	chunkSize := max(p.fault.Chunk, 1)
	var pendIDs []int
	var pendText strings.Builder
	held := ""
	sent := 0
	buf := make([]byte, 0, 512)

	write := func(ch chunkJSON) bool {
		b, _ := json.Marshal(ch)
		buf = append(buf[:0], "data: "...)
		buf = append(buf, b...)
		buf = append(buf, "\n\n"...)
		if _, err := w.Write(buf); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	emit := func(finish *string, stopReason json.RawMessage) bool {
		ch := chunkJSON{RequestID: p.engineID, OutputMode: "tokens", Choices: []choiceJSON{{
			FinishReason: finish, StopReason: stopReason, TokenIDs: pendIDs, Logprobs: jsonNull,
		}}}
		if ch.Choices[0].StopReason == nil {
			ch.Choices[0].StopReason = jsonNull
		}
		if ch.Choices[0].TokenIDs == nil {
			ch.Choices[0].TokenIDs = []int{}
		}
		if p.textMode {
			t := pendText.String()
			ch.OutputMode = "text"
			ch.Choices[0].Text = &t
		}
		pendIDs = nil
		pendText.Reset()
		return write(ch)
	}

	for i := 0; i < p.max; i++ {
		if i == p.fault.At && p.fault.Kind != "" && p.fault.Kind != "5xx" {
			s.Stats.Faults.Add(1)
			switch p.fault.Kind {
			case "die":
				if len(pendIDs) > 0 {
					emit(nil, nil)
				}
				reset(w)
				return
			case "cut":
				half := s.nextEventBytes(p, st, seq, i, ds)
				if _, err := w.Write(half[:len(half)/2]); err == nil && flusher != nil {
					flusher.Flush()
				}
				reset(w)
				return
			case "eof":
				if len(pendIDs) > 0 {
					emit(nil, nil)
				}
				return
			case "err":
				if len(pendIDs) > 0 {
					emit(nil, nil)
				}
				b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": "injected engine error", "type": "InternalServerError", "param": nil, "code": 500}})
				buf = append(append(append(buf[:0], "data: "...), b...), "\n\ndata: [DONE]\n\n"...)
				_, _ = w.Write(buf)
				return
			case "abort":
				if len(pendIDs) > 0 {
					emit(nil, nil)
				}
				s.finishAborted(w, flusher, p, sent)
				return
			case "stall":
				if len(pendIDs) > 0 && !emit(nil, nil) {
					s.Stats.Disconnected.Add(1)
					return
				}
				if aborted, gone := sleep(ctx, rn, p.fault.Pause); gone {
					s.Stats.Disconnected.Add(1)
					return
				} else if aborted {
					s.Stats.Aborted.Add(1)
					s.finishAborted(w, flusher, p, sent)
					return
				}
			}
		}
		if i > 0 {
			if aborted, gone := sleep(ctx, rn, s.itl(p, i)); gone {
				s.Stats.Disconnected.Add(1)
				return
			} else if aborted {
				s.Stats.Aborted.Add(1)
				if len(pendIDs) > 0 {
					emit(nil, nil)
				}
				s.finishAborted(w, flusher, p, sent)
				return
			}
		} else if aborted, gone := sleep(ctx, rn, 0); gone {
			s.Stats.Disconnected.Add(1)
			return
		} else if aborted {
			s.Stats.Aborted.Add(1)
			s.finishAborted(w, flusher, p, sent)
			return
		}

		tok := s.Model.Next(st, i, p.sampling)
		if i < p.min && (tok == TokIMEnd || p.sampling.StopTokens[tok]) {
			tok = s.Vocab.gen[splitmix(uint64(i)^st.h)%uint64(len(s.Vocab.gen))]
		}
		sent++
		s.Stats.Tokens.Add(1)
		pendIDs = append(pendIDs, tok)
		if p.textMode {
			t := ds.Step(tok)
			if p.fault.Holdback {
				t, held = held, t
			}
			pendText.WriteString(t)
		}
		var finish *string
		var stopReason json.RawMessage
		if tok == TokIMEnd {
			f := "stop"
			finish = &f
		} else if p.sampling.StopTokens[tok] {
			f := "stop"
			finish = &f
			stopReason = json.RawMessage(strconv.Itoa(tok))
		} else if i == p.max-1 {
			f := "length"
			finish = &f
		}
		if finish != nil && p.fault.Holdback {
			pendText.WriteString(held)
			held = ""
		}
		st = st.Push(tok)
		seq = append(seq, tok)
		if finish != nil || len(pendIDs) >= chunkSize {
			if !emit(finish, stopReason) {
				s.Stats.Disconnected.Add(1)
				return
			}
		}
		if finish != nil {
			break
		}
	}
	s.cache.Insert(seq)
	s.finishUsage(w, flusher, p, sent)
}

func (s *Server) finishAborted(w http.ResponseWriter, flusher http.Flusher, p *genParams, sent int) {
	f := "abort"
	ch := chunkJSON{RequestID: p.engineID, OutputMode: "tokens", Choices: []choiceJSON{{FinishReason: &f, StopReason: jsonNull, TokenIDs: []int{}, Logprobs: jsonNull}}}
	if p.textMode {
		e := ""
		ch.OutputMode = "text"
		ch.Choices[0].Text = &e
	}
	b, _ := json.Marshal(ch)
	_, _ = w.Write(append(append([]byte("data: "), b...), "\n\n"...))
	s.finishUsage(w, flusher, p, sent)
}

func (s *Server) finishUsage(w http.ResponseWriter, flusher http.Flusher, p *genParams, sent int) {
	var out []byte
	if p.usage {
		mode := "tokens"
		if p.textMode {
			mode = "text"
		}
		b, _ := json.Marshal(chunkJSON{RequestID: p.engineID, OutputMode: mode, Choices: []choiceJSON{},
			Usage: &usageJSON{PromptTokens: len(p.prompt), CompletionTokens: sent, TotalTokens: len(p.prompt) + sent}})
		out = append(append(append(out, "data: "...), b...), "\n\n"...)
	}
	out = append(out, "data: [DONE]\n\n"...)
	_, _ = w.Write(out)
	if flusher != nil {
		flusher.Flush()
	}
}

// nextEventBytes renders the event that token i would produce, so a "cut"
// fault can send half of it.
func (s *Server) nextEventBytes(p *genParams, st State, seq []int, i int, ds *DecodeStream) []byte {
	tok := s.Model.Next(st, i, p.sampling)
	ch := chunkJSON{RequestID: p.engineID, OutputMode: "tokens", Choices: []choiceJSON{{TokenIDs: []int{tok}, StopReason: jsonNull, Logprobs: jsonNull}}}
	if p.textMode {
		t := ds.Step(tok)
		ch.OutputMode = "text"
		ch.Choices[0].Text = &t
	}
	b, _ := json.Marshal(ch)
	return append(append([]byte("data: "), b...), "\n\n"...)
}

// reset aborts the TCP connection with RST, the way a killed process's
// socket looks to the peer.
func reset(w http.ResponseWriter) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	conn, _, err := http.NewResponseController(w).Hijack()
	if err != nil {
		panic(http.ErrAbortHandler)
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		_ = tc.SetLinger(0)
	}
	_ = conn.Close()
}
