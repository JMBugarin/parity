package detsim

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"syscall"
	"testing"
	"time"
)

func genBody(t *testing.T, s *Server, id string, max int) []byte {
	t.Helper()
	prompt := s.Vocab.ChatPrompt([]Message{{Role: "user", Content: "hooks 🦊"}})
	b, _ := json.Marshal(map[string]any{
		"request_id": id, "token_ids": prompt, "stream": true, "output_mode": "text",
		"stream_options":  map[string]any{"include_usage": true},
		"sampling_params": map[string]any{"max_tokens": max, "temperature": 0, "ignore_eos": true},
	})
	return b
}

func do(t *testing.T, url string, body []byte, fault string) (*http.Response, []byte, error) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url+"/inference/v1/generate", bytes.NewReader(body))
	if fault != "" {
		req.Header.Set("X-Detsim-Fault", fault)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return resp, b, err
}

func TestSameRequestSameBytes(t *testing.T) {
	s := New(Config{Seed: 5, EOSPeriod: 50})
	ts := httptest.NewServer(s)
	defer ts.Close()
	_, a, err := do(t, ts.URL, genBody(t, s, "same", 120), "")
	if err != nil {
		t.Fatal(err)
	}
	_, b, err := do(t, ts.URL, genBody(t, s, "same", 120), "")
	if err != nil || !bytes.Equal(a, b) {
		t.Fatalf("two identical requests gave different bytes (%v)", err)
	}
	if !bytes.HasSuffix(a, []byte("data: [DONE]\n\n")) {
		t.Fatal("no [DONE]")
	}
}

func TestFaultHooks(t *testing.T) {
	s := New(Config{Seed: 5})
	ts := httptest.NewServer(s)
	defer ts.Close()
	events := func(b []byte) int { return bytes.Count(b, []byte("\"token_ids\":[")) }

	if resp, _, _ := do(t, ts.URL, genBody(t, s, "f5xx", 50), "1:5xx"); resp.StatusCode != 503 {
		t.Fatalf("5xx hook: %d", resp.StatusCode)
	}
	_, b, err := do(t, ts.URL, genBody(t, s, "fdie", 50), "1:die@7")
	if err == nil || !errors.Is(err, syscall.ECONNRESET) && !strings.Contains(err.Error(), "reset") && !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("die hook: err %v", err)
	}
	if n := events(b); n != 7 {
		t.Fatalf("die@7 sent %d token events", n)
	}
	_, b, err = do(t, ts.URL, genBody(t, s, "fcut", 50), "1:cut@4")
	if err == nil || bytes.HasSuffix(b, []byte("\n\n")) {
		t.Fatalf("cut hook should end inside an event: err %v tail %q", err, b[max(0, len(b)-20):])
	}
	_, b, err = do(t, ts.URL, genBody(t, s, "feof", 50), "1:eof@9")
	if err != nil || events(b) != 9 || bytes.Contains(b, []byte("[DONE]")) {
		t.Fatalf("eof hook: err %v events %d", err, events(b))
	}
	_, b, _ = do(t, ts.URL, genBody(t, s, "ferr", 50), "1:err@3")
	if !bytes.Contains(b, []byte(`"error"`)) || !bytes.Contains(b, []byte("[DONE]")) {
		t.Fatal("err hook: no error event")
	}
	_, b, _ = do(t, ts.URL, genBody(t, s, "fabort", 50), "1:abort@5")
	if !bytes.Contains(b, []byte(`"finish_reason":"abort"`)) {
		t.Fatal("abort hook: no abort finish")
	}
	_, b, _ = do(t, ts.URL, genBody(t, s, "fchunk", 30), "*:chunk=4")
	if n := bytes.Count(b, []byte(`"finish_reason":null`)); n != 7 {
		t.Fatalf("chunk=4 over 30 tokens: %d open chunks", n)
	}
	// Attempt 2 of the same plan is unaffected by attempt 1's fault.
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/inference/v1/generate", bytes.NewReader(genBody(t, s, "fa2", 20)))
	req.Header.Set("X-Detsim-Fault", "1:die@3")
	req.Header.Set("X-Parity-Attempt", "2")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Contains(b, []byte("[DONE]")) {
		t.Fatal("attempt 2 should not get attempt 1's fault")
	}
}

func TestStallThenAbort(t *testing.T) {
	s := New(Config{Seed: 5})
	ts := httptest.NewServer(s)
	defer ts.Close()
	done := make(chan []byte, 1)
	go func() {
		_, b, _ := do(t, ts.URL, genBody(t, s, "fstall", 100), "1:stall@10/5s")
		done <- b
	}()
	time.Sleep(100 * time.Millisecond)
	resp, err := http.Post(ts.URL+"/inference/v1/abort_requests", "application/json", strings.NewReader(`{"request_ids":["generate-tokens-fstall"]}`))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("abort: %v", err)
	}
	resp.Body.Close()
	select {
	case b := <-done:
		if !bytes.Contains(b, []byte(`"finish_reason":"abort"`)) {
			t.Fatal("stalled request did not finish with abort")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("abort did not end the stall")
	}
	if s.Stats.Aborted.Load() != 1 {
		t.Fatalf("aborted = %d", s.Stats.Aborted.Load())
	}
}

func TestHoldbackDelaysText(t *testing.T) {
	s := New(Config{Seed: 5})
	ts := httptest.NewServer(s)
	defer ts.Close()
	_, plain, _ := do(t, ts.URL, genBody(t, s, "h1", 40), "")
	_, held, _ := do(t, ts.URL, genBody(t, s, "h2", 40), "*:holdback")
	text := func(b []byte) (first, all string) {
		for _, line := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(line, "data: {") {
				continue
			}
			var c struct {
				Choices []struct {
					Text string `json:"text"`
				} `json:"choices"`
			}
			_ = json.Unmarshal([]byte(line[6:]), &c)
			for _, ch := range c.Choices {
				if first == "" && ch.Text != "" {
					first = ch.Text
				}
				all += ch.Text
			}
		}
		return
	}
	_, a := text(plain)
	_, b := text(held)
	if a != b {
		t.Fatal("holdback changed the final text")
	}
	if !bytes.Contains(held, []byte(`"text":""`)) {
		t.Fatal("holdback did not hold the first token's text")
	}
}

func TestRenderEndpoint(t *testing.T) {
	s := New(Config{Seed: 5, MaxModelLen: 4096})
	ts := httptest.NewServer(s)
	defer ts.Close()
	resp, err := http.Post(ts.URL+"/v1/chat/completions/render", "application/json",
		strings.NewReader(`{"model":"detsim","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"temperature":0,"stop":"STOP"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var r struct {
		TokenIDs       []int          `json:"token_ids"`
		SamplingParams map[string]any `json:"sampling_params"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		t.Fatal(err)
	}
	want := s.Vocab.ChatPrompt([]Message{{Role: "user", Content: "hi"}})
	if !equal(r.TokenIDs, want) || r.SamplingParams["max_tokens"].(float64) != 9 {
		t.Fatalf("render %+v", r)
	}
	if stops, _ := r.SamplingParams["stop"].([]any); len(stops) != 1 || stops[0] != "STOP" {
		t.Fatalf("stop %v", r.SamplingParams["stop"])
	}
}
