package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"syscall"
	"time"

	"github.com/JMBugarin/parity/internal/sse"
)

// Kind classifies why an upstream call or stream failed.
type Kind string

const (
	KindConnect   Kind = "connect"
	KindStatus    Kind = "status"
	KindReset     Kind = "reset"
	KindEOF       Kind = "eof"
	KindTruncated Kind = "truncated"
	KindEvent     Kind = "error_event"
	KindDecode    Kind = "decode"
	KindCanceled  Kind = "canceled"
)

// Error is an upstream failure with enough detail to decide whether to retry.
type Error struct {
	Kind   Kind
	Status int
	Msg    string
	Err    error
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("upstream ")
	b.WriteString(string(e.Kind))
	if e.Status != 0 {
		fmt.Fprintf(&b, " %d", e.Status)
	}
	if e.Msg != "" {
		b.WriteString(": ")
		b.WriteString(e.Msg)
	}
	if e.Err != nil {
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	}
	return b.String()
}

func (e *Error) Unwrap() error { return e.Err }

// Retryable reports whether another replica could serve the request. Client
// errors (4xx other than 408/429) are not retryable: the request itself is
// wrong and every replica would reject it.
func (e *Error) Retryable() bool {
	switch e.Kind {
	case KindStatus:
		return e.Status >= 500 || e.Status == 408 || e.Status == 429
	case KindCanceled, KindDecode:
		return false
	}
	return true
}

// ErrDone is returned by Stream.Next after the [DONE] sentinel.
var ErrDone = errors.New("upstream: done")

type Options struct {
	DialTimeout   time.Duration
	HeaderTimeout time.Duration
	// CallTimeout bounds render, abort and detokenize.
	CallTimeout  time.Duration
	MaxEventSize int
	Transport    http.RoundTripper
}

type Client struct {
	stream  *http.Client
	call    *http.Client
	maxEv   int
	timeout time.Duration
}

func NewTransport(o Options) *http.Transport {
	if o.DialTimeout == 0 {
		o.DialTimeout = 2 * time.Second
	}
	if o.HeaderTimeout == 0 {
		o.HeaderTimeout = 15 * time.Second
	}
	return &http.Transport{
		DialContext:           (&net.Dialer{Timeout: o.DialTimeout, KeepAlive: 15 * time.Second}).DialContext,
		MaxIdleConns:          2048,
		MaxIdleConnsPerHost:   512,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: o.HeaderTimeout,
		DisableCompression:    true,
	}
}

func New(o Options) *Client {
	tr := o.Transport
	if tr == nil {
		tr = NewTransport(o)
	}
	if o.CallTimeout == 0 {
		o.CallTimeout = 5 * time.Second
	}
	return &Client{
		stream:  &http.Client{Transport: tr},
		call:    &http.Client{Transport: tr, Timeout: o.CallTimeout},
		maxEv:   o.MaxEventSize,
		timeout: o.CallTimeout,
	}
}

func classify(err error) *Error {
	var ue *Error
	if errors.As(err, &ue) {
		return ue
	}
	switch {
	case errors.Is(err, context.Canceled):
		return &Error{Kind: KindCanceled, Err: err}
	case errors.Is(err, sse.ErrTruncated):
		return &Error{Kind: KindTruncated, Err: err}
	case errors.Is(err, io.ErrUnexpectedEOF), errors.Is(err, io.EOF):
		return &Error{Kind: KindEOF, Err: err}
	case errors.Is(err, syscall.ECONNRESET), errors.Is(err, syscall.EPIPE), errors.Is(err, net.ErrClosed):
		return &Error{Kind: KindReset, Err: err}
	}
	var op *net.OpError
	if errors.As(err, &op) && op.Op == "dial" {
		return &Error{Kind: KindConnect, Err: err}
	}
	return &Error{Kind: KindReset, Err: err}
}

func (c *Client) post(ctx context.Context, hc *http.Client, url string, body []byte, hdr http.Header) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, &Error{Kind: KindDecode, Err: err}
	}
	for k, v := range hdr {
		req.Header[k] = v
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := hc.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, &Error{Kind: KindCanceled, Err: ctx.Err()}
		}
		e := classify(err)
		if e.Kind == KindReset || e.Kind == KindEOF {
			var op *net.OpError
			if errors.As(err, &op) && op.Op == "dial" {
				e.Kind = KindConnect
			}
		}
		return nil, e
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		resp.Body.Close()
		return nil, &Error{Kind: KindStatus, Status: resp.StatusCode, Msg: ErrorMessage(b)}
	}
	return resp, nil
}

// Render applies the chat template on a replica and returns the prompt token
// IDs and resolved sampling parameters.
func (c *Client) Render(ctx context.Context, base string, chatBody []byte, hdr http.Header) (*Rendered, error) {
	resp, err := c.post(ctx, c.call, base+"/v1/chat/completions/render", chatBody, hdr)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, classify(err)
	}
	r, err := ParseRendered(b)
	if err != nil {
		return nil, &Error{Kind: KindDecode, Err: err}
	}
	return r, nil
}

// Stream is an open /inference/v1/generate response.
type Stream struct {
	resp *http.Response
	rd   *sse.Reader
}

// Generate starts a streaming generation. The caller must Close the stream.
func (c *Client) Generate(ctx context.Context, base string, body []byte, hdr http.Header) (*Stream, error) {
	resp, err := c.post(ctx, c.stream, base+"/inference/v1/generate", body, hdr)
	if err != nil {
		return nil, err
	}
	return &Stream{resp: resp, rd: sse.NewReader(resp.Body, c.maxEv)}, nil
}

// Next returns the next chunk, ErrDone after [DONE], or an *Error.
func (s *Stream) Next() (*Chunk, error) {
	ev, err := s.rd.Next()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return nil, &Error{Kind: KindEOF, Msg: "stream ended without [DONE]"}
		}
		return nil, classify(err)
	}
	if bytes.Equal(ev.Data, []byte("[DONE]")) {
		return nil, ErrDone
	}
	var ch Chunk
	if err := json.Unmarshal(ev.Data, &ch); err != nil {
		return nil, &Error{Kind: KindDecode, Err: err}
	}
	if len(ch.Error) > 0 && string(ch.Error) != "null" {
		return nil, &Error{Kind: KindEvent, Msg: ErrorMessage(ev.Data)}
	}
	return &ch, nil
}

func (s *Stream) Close() error { return s.resp.Body.Close() }

// Abort asks a replica to stop generating the given engine request IDs.
func (c *Client) Abort(ctx context.Context, base string, ids ...string) error {
	body, _ := json.Marshal(map[string][]string{"request_ids": ids})
	resp, err := c.post(ctx, c.call, base+"/inference/v1/abort_requests", body, nil)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, resp.Body)
	return resp.Body.Close()
}

// Detokenize decodes token IDs with the replica's tokenizer. vLLM's endpoint
// does not skip special tokens.
func (c *Client) Detokenize(ctx context.Context, base, model string, ids []int) (string, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "tokens": ids})
	resp, err := c.post(ctx, c.call, base+"/detokenize", body, nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var out struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&out); err != nil {
		return "", &Error{Kind: KindDecode, Err: err}
	}
	return out.Prompt, nil
}

// Health probes GET /health.
func (c *Client) Health(ctx context.Context, base string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.call.Do(req)
	if err != nil {
		return classify(err)
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return &Error{Kind: KindStatus, Status: resp.StatusCode}
	}
	return nil
}

// EngineRequestID is the ID vLLM assigns to a generate request, which is
// what /inference/v1/abort_requests expects.
func EngineRequestID(id string) string { return "generate-tokens-" + id }

// Drain reads the rest of the body, up to a small limit, so the connection
// can be reused.
func (s *Stream) Drain() {
	_, _ = io.Copy(io.Discard, io.LimitReader(s.resp.Body, 64<<10))
}
