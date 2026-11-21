package detsim

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	v := NewVocab()
	for _, s := range []string{"", "hello world", "日本語 🦊 café", "STOP<|end|>\n\n###", "\x00\xff raw", "the quick brown fox"} {
		ids := v.Encode(s)
		if got := string(v.Bytes(ids, false)); got != s {
			t.Fatalf("round trip %q -> %q", s, got)
		}
	}
}

func TestLossyMatchesMaximalSubparts(t *testing.T) {
	cases := map[string]string{
		"a\xF0\x9F\xA6":    "a�",
		"\xF0\x9F\xA6\x8A": "🦊",
		"\xE6\x97":         "\uFFFD",
		"\xC3\x28":         "�(",
		"\x80\x80":         "��",
		"\xED\xA0\x80":     "���",
		"\xF4\x90\x80\x80": "����",
	}
	for in, want := range cases {
		if got := Lossy([]byte(in)); got != want {
			t.Errorf("Lossy(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestModelDeterministicFromPrefix(t *testing.T) {
	v := NewVocab()
	m := &Model{Vocab: v, Seed: 7}
	prompt := v.ChatPrompt([]Message{{Role: "user", Content: "tell me a story"}})
	full, _ := m.Run(prompt, 300, Sampling{IgnoreEOS: true})
	for _, k := range []int{1, 17, 137, 299} {
		rest, _ := m.Run(append(append([]int(nil), prompt...), full[:k]...), 300-k, Sampling{IgnoreEOS: true})
		for i := range rest {
			if rest[i] != full[k+i] {
				t.Fatalf("k=%d: token %d differs", k, i)
			}
		}
	}
	other := &Model{Vocab: v, Seed: 8}
	if o, _ := other.Run(prompt, 50, Sampling{IgnoreEOS: true}); equal(o, full[:50]) {
		t.Fatal("different seeds produced the same output")
	}
}

func TestSampledResumeDiffers(t *testing.T) {
	v := NewVocab()
	m := &Model{Vocab: v, Seed: 7}
	prompt := v.ChatPrompt([]Message{{Role: "user", Content: "x"}})
	sp := Sampling{Temperature: 1, Seed: 42, IgnoreEOS: true}
	full, _ := m.Run(prompt, 200, sp)
	again, _ := m.Run(prompt, 200, sp)
	if !equal(full, again) {
		t.Fatal("seeded sampling is not reproducible")
	}
	rest, _ := m.Run(append(append([]int(nil), prompt...), full[:50]...), 150, sp)
	if equal(rest, full[50:]) {
		t.Fatal("resumed sampled stream matched; the RNG should restart at the splice")
	}
}

func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestDecodeStreamReproducesVLLM mirrors docs/evidence/M0: resuming after
// the first two of the three tokens of " 🦊" makes vLLM v0.34.0 emit the
// whole prompt with the character.
func TestDecodeStreamReproducesVLLM(t *testing.T) {
	v := NewVocab()
	prompt := v.ChatPrompt([]Message{{Role: "user", Content: "hi"}})
	fox := v.Encode(" \xF0\x9F")
	if len(fox) != 1 || v.Tokens[fox[0]].Next < 0 {
		t.Fatalf("fox chain start not found: %v", fox)
	}
	a := fox[0]
	b := v.Tokens[a].Next
	c := v.Tokens[b].Next

	ds := NewDecodeStream(v, prompt, true, false)
	if got := ds.Step(a) + "|" + ds.Step(b) + "|" + ds.Step(c); got != "||"+" 🦊" {
		t.Fatalf("clean prompt: %q", got)
	}

	resumed := append(append([]int(nil), prompt...), a, b)
	ds = NewDecodeStream(v, resumed, true, false)
	first := ds.Step(c)
	if !strings.HasPrefix(first, "user\nhi") || !strings.HasSuffix(first, " 🦊") {
		t.Fatalf("vLLM-mode resume after a split character: %q", first)
	}
	if next := ds.Step(v.Encode(" the")[0]); next != " the" {
		t.Fatalf("after the glitch the stream should be back in sync, got %q", next)
	}

	ds = NewDecodeStream(v, resumed, true, true)
	if got := ds.Step(c); got != " 🦊" {
		t.Fatalf("fixed mode: %q", got)
	}
}

func TestDecodeStreamConcatenatesToDecode(t *testing.T) {
	v := NewVocab()
	m := &Model{Vocab: v, Seed: 3}
	prompt := v.ChatPrompt([]Message{{Role: "user", Content: "go"}})
	out, _ := m.Run(prompt, 2000, Sampling{IgnoreEOS: true})
	ds := NewDecodeStream(v, prompt, true, false)
	var b strings.Builder
	for _, tok := range out {
		b.WriteString(ds.Step(tok))
	}
	want := strings.TrimSuffix(v.Decode(out, true), "\uFFFD")
	if b.String() != want {
		t.Fatalf("incremental text differs from batch decode")
	}
	if !utf8.ValidString(b.String()) {
		t.Fatal("invalid UTF-8 emitted")
	}
}

func TestGenerateEndpoint(t *testing.T) {
	s := New(Config{Seed: 1})
	ts := httptest.NewServer(s)
	defer ts.Close()
	prompt := s.Vocab.ChatPrompt([]Message{{Role: "user", Content: "hello"}})
	body, _ := json.Marshal(map[string]any{
		"request_id": "t1", "token_ids": prompt, "stream": true, "output_mode": "text",
		"stream_options":  map[string]any{"include_usage": true},
		"sampling_params": map[string]any{"max_tokens": 40, "temperature": 0, "ignore_eos": true},
	})
	resp, err := http.Post(ts.URL+"/inference/v1/generate", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	var ids []int
	var text strings.Builder
	done := false
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			done = true
			break
		}
		var ch struct {
			Choices []struct {
				TokenIDs []int  `json:"token_ids"`
				Text     string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			t.Fatal(err)
		}
		for _, c := range ch.Choices {
			ids = append(ids, c.TokenIDs...)
			text.WriteString(c.Text)
		}
	}
	want, _ := s.Model.Run(prompt, 40, Sampling{IgnoreEOS: true})
	if !done || !equal(ids, want) {
		t.Fatalf("done=%v ids=%v want %v", done, ids, want)
	}
	if text.String() != strings.TrimSuffix(s.Vocab.Decode(want, true), "\uFFFD") {
		t.Fatalf("text %q", text.String())
	}
}

func TestParsePlan(t *testing.T) {
	p, err := ParsePlan("1:die@137;2:stall@20/300ms,drip/2ms;*:chunk=3,holdback")
	if err != nil {
		t.Fatal(err)
	}
	if p.For(1).Kind != "die" || p.For(1).At != 137 || p.For(2).Pause.Milliseconds() != 300 || p.For(5).Chunk != 3 || !p.For(5).Holdback {
		t.Fatalf("plan %+v", p)
	}
	q, err := ParsePlan(p.String())
	if err != nil || q.String() != p.String() {
		t.Fatalf("round trip %q -> %q (%v)", p.String(), q.String(), err)
	}
	if _, err := ParsePlan("1:explode@3"); err == nil {
		t.Fatal("unknown fault accepted")
	}
}
