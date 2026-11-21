package upstream

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestParseRenderedFromVLLM(t *testing.T) {
	b, err := os.ReadFile("testdata/render.json")
	if err != nil {
		t.Fatal(err)
	}
	r, err := ParseRendered(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.TokenIDs) != 16 || r.Model != "qwen3-0.6b" {
		t.Fatalf("ids %d model %q", len(r.TokenIDs), r.Model)
	}
	if n, ok := r.Int("max_tokens"); !ok || n != 24 {
		t.Fatalf("max_tokens %d %v", n, ok)
	}
	if s := r.Stops(); len(s) != 1 || s[0] != "\n\n" {
		t.Fatalf("stops %q", s)
	}
	if f, ok := r.Float("temperature"); !ok || f != 0 {
		t.Fatalf("temperature %v", f)
	}
}

func TestGenerateBodyMovesStopsToGateway(t *testing.T) {
	b, _ := os.ReadFile("testdata/render.json")
	r, _ := ParseRendered(b)
	body, err := r.GenerateBody(Attempt{RequestID: "abc-a2", TokenIDs: []int{1, 2, 3}, MaxTokens: 7, MinTokens: 0, OutputMode: "text"})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		RequestID      string                     `json:"request_id"`
		TokenIDs       []int                      `json:"token_ids"`
		Stream         bool                       `json:"stream"`
		OutputMode     string                     `json:"output_mode"`
		SamplingParams map[string]json.RawMessage `json:"sampling_params"`
		StreamOptions  map[string]bool            `json:"stream_options"`
		Model          string                     `json:"model"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got.RequestID != "abc-a2" || len(got.TokenIDs) != 3 || !got.Stream || got.OutputMode != "text" || got.Model != "qwen3-0.6b" {
		t.Fatalf("body %+v", got)
	}
	sp := got.SamplingParams
	if _, ok := sp["stop"]; ok {
		t.Fatal("stop strings must not reach the engine")
	}
	if _, ok := sp["output_kind"]; ok {
		t.Fatal("output_kind must be left to the generate handler")
	}
	if string(sp["max_tokens"]) != "7" || string(sp["output_text_buffer_length"]) != "0" || string(sp["temperature"]) != "0.0" {
		t.Fatalf("sampling params %s %s %s", sp["max_tokens"], sp["output_text_buffer_length"], sp["temperature"])
	}
	if !got.StreamOptions["include_usage"] {
		t.Fatal("usage not requested")
	}
}

func serveFile(t *testing.T, path string) *httptest.Server {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(raw)
	}))
}

func TestStreamRecordedText(t *testing.T) {
	ts := serveFile(t, "../sse/testdata/vllm-generate-text.sse")
	defer ts.Close()
	c := New(Options{})
	st, err := c.Generate(context.Background(), ts.URL, []byte(`{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var text strings.Builder
	var ids []int
	finish := ""
	var usage *Usage
	for {
		ch, err := st.Next()
		if errors.Is(err, ErrDone) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if ch.RequestID != "generate-tokens-parity-m0-1" {
			t.Fatalf("request id %q", ch.RequestID)
		}
		if ch.Usage != nil {
			usage = ch.Usage
		}
		for _, c := range ch.Choices {
			ids = append(ids, c.TokenIDs...)
			if c.Text != nil {
				text.WriteString(*c.Text)
			}
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
	}
	if finish != "stop" || usage == nil || usage.CompletionTokens != len(ids) || text.Len() == 0 {
		t.Fatalf("finish %q usage %+v ids %d text %q", finish, usage, len(ids), text.String())
	}
}

func TestStreamErrors(t *testing.T) {
	cases := map[string]struct {
		body string
		kind Kind
	}{
		"eof before done": {"data: {\"choices\":[]}\n\n", KindEOF},
		"truncated":       {"data: {\"choices\":[]}\n\ndata: {\"cho", KindTruncated},
		"error event":     {"data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\ndata: [DONE]\n\n", KindEvent},
		"bad json":        {"data: {nope\n\n", KindDecode},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.WriteString(w, tc.body)
			}))
			defer ts.Close()
			st, err := New(Options{}).Generate(context.Background(), ts.URL, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer st.Close()
			var last error
			for {
				_, err := st.Next()
				if err != nil {
					last = err
					break
				}
			}
			var ue *Error
			if !errors.As(last, &ue) || ue.Kind != tc.kind {
				t.Fatalf("got %v, want kind %s", last, tc.kind)
			}
		})
	}
}

func TestStatusAndConnectErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"draining","code":503}}`)
	}))
	c := New(Options{DialTimeout: 200 * time.Millisecond})
	_, err := c.Generate(context.Background(), ts.URL, nil, nil)
	var ue *Error
	if !errors.As(err, &ue) || ue.Kind != KindStatus || ue.Status != 503 || ue.Msg != "draining" || !ue.Retryable() {
		t.Fatalf("got %#v", err)
	}
	url := ts.URL
	ts.Close()
	_, err = c.Generate(context.Background(), url, nil, nil)
	if !errors.As(err, &ue) || ue.Kind != KindConnect || !ue.Retryable() {
		t.Fatalf("closed server: %#v", err)
	}
	if (&Error{Kind: KindStatus, Status: 400}).Retryable() {
		t.Fatal("400 must not be retried")
	}
}

func TestAbortAndDetokenize(t *testing.T) {
	var gotAbort []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/inference/v1/abort_requests":
			var v struct {
				RequestIDs []string `json:"request_ids"`
			}
			_ = json.NewDecoder(r.Body).Decode(&v)
			gotAbort = v.RequestIDs
		case "/detokenize":
			_, _ = io.WriteString(w, `{"prompt":"Three colors could"}`)
		}
	}))
	defer ts.Close()
	c := New(Options{})
	if err := c.Abort(context.Background(), ts.URL, EngineRequestID("x-a1")); err != nil {
		t.Fatal(err)
	}
	if len(gotAbort) != 1 || gotAbort[0] != "generate-tokens-x-a1" {
		t.Fatalf("abort ids %q", gotAbort)
	}
	s, err := c.Detokenize(context.Background(), ts.URL, "m", []int{19641, 7987, 1410})
	if err != nil || s != "Three colors could" {
		t.Fatalf("detokenize %q %v", s, err)
	}
}

func TestFaultTransportCutsAfterEvents(t *testing.T) {
	ts := serveFile(t, "../sse/testdata/vllm-generate-text.sse")
	defer ts.Close()
	c := New(Options{Transport: &FaultTransport{Base: NewTransport(Options{})}})
	hdr := http.Header{"X-Parity-Fault": {"1:die@3"}, "X-Parity-Attempt": {"1"}}
	st, err := c.Generate(context.Background(), ts.URL, []byte(`{}`), hdr)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	n := 0
	for {
		_, err := st.Next()
		if err != nil {
			var ue *Error
			if !errors.As(err, &ue) || ue.Kind != KindReset {
				t.Fatalf("got %v", err)
			}
			break
		}
		n++
	}
	if n != 3 {
		t.Fatalf("events before the cut: %d", n)
	}
}
