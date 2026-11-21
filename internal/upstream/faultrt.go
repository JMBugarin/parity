package upstream

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// FaultTransport cuts or stalls generate streams on the gateway side, for
// experiments against real engines that cannot be told to fail. The plan
// comes from the X-Parity-Fault request header, in detsim's syntax but with
// only die@k (reset after k events) and stall@k/d.
type FaultTransport struct {
	Base http.RoundTripper
}

func (t *FaultTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.Base.RoundTrip(req)
	spec := req.Header.Get("X-Parity-Fault")
	if err != nil || spec == "" || !strings.HasSuffix(req.URL.Path, "/inference/v1/generate") {
		return resp, err
	}
	attempt, _ := strconv.Atoi(req.Header.Get("X-Parity-Attempt"))
	kind, at, pause, err := faultFor(spec, attempt)
	if err != nil || kind == "" {
		return resp, nil
	}
	resp.Body = &faultBody{rc: resp.Body, kind: kind, at: at, pause: pause, done: req.Context().Done()}
	return resp, nil
}

func faultFor(spec string, attempt int) (kind string, at int, pause time.Duration, err error) {
	for _, part := range strings.Split(spec, ";") {
		key, f, ok := strings.Cut(strings.TrimSpace(part), ":")
		if !ok || (key != "*" && key != strconv.Itoa(attempt)) {
			continue
		}
		k, rest, ok := strings.Cut(f, "@")
		if !ok {
			return "", 0, 0, fmt.Errorf("bad fault %q", f)
		}
		pos, d, _ := strings.Cut(rest, "/")
		at, err = strconv.Atoi(pos)
		if err != nil {
			return "", 0, 0, err
		}
		if k == "stall" {
			if pause, err = time.ParseDuration(d); err != nil {
				return "", 0, 0, err
			}
		}
		return k, at, pause, nil
	}
	return "", 0, 0, nil
}

type faultBody struct {
	rc    io.ReadCloser
	kind  string
	at    int
	pause time.Duration
	done  <-chan struct{}

	events  int
	prevNL  bool
	pending []byte
	fired   bool
}

func (b *faultBody) Read(p []byte) (int, error) {
	if b.fired && b.kind == "die" {
		return 0, syscall.ECONNRESET
	}
	if len(b.pending) == 0 {
		buf := make([]byte, len(p))
		n, err := b.rc.Read(buf)
		b.pending = buf[:n]
		if n == 0 {
			return 0, err
		}
	}
	// Pass bytes through up to the end of event number b.at.
	if !b.fired && b.events < b.at {
		out := b.pending
		cut := len(out)
		for i := 0; i < len(out); i++ {
			if out[i] != '\n' {
				b.prevNL = false
				continue
			}
			if !b.prevNL {
				b.prevNL = true
				continue
			}
			b.prevNL = false
			b.events++
			if b.events == b.at {
				cut = i + 1
				break
			}
		}
		n := copy(p, out[:cut])
		b.pending = b.pending[n:]
		return n, nil
	}
	if !b.fired {
		b.fired = true
		switch b.kind {
		case "die":
			b.rc.Close()
			return 0, syscall.ECONNRESET
		case "stall":
			t := time.NewTimer(b.pause)
			select {
			case <-t.C:
			case <-b.done:
				t.Stop()
				return 0, io.ErrUnexpectedEOF
			}
		}
	}
	n := copy(p, b.pending)
	b.pending = b.pending[n:]
	return n, nil
}

func (b *faultBody) Close() error { return b.rc.Close() }
