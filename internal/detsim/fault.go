package detsim

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Fault is what a replica does to one attempt of a request.
type Fault struct {
	// Kind is one of: die (TCP reset), cut (half an SSE event, then reset),
	// eof (clean close without finish), stall (pause, then continue unless
	// aborted), err (an SSE error event), abort (finish_reason "abort"),
	// 5xx (HTTP 503 before any token).
	Kind  string
	At    int
	Pause time.Duration
	// Drip delays every token; SlowFirst delays the first one.
	Drip      time.Duration
	SlowFirst time.Duration
	// Holdback releases each token's text one chunk late, like an engine
	// that buffers text for stop-string matching.
	Holdback bool
	// Chunk groups this many tokens per SSE event, as vLLM does when the
	// reader falls behind.
	Chunk int
}

// Plan maps attempt numbers (1-based) to faults. Attempt 0 applies to every
// attempt without its own entry.
type Plan map[int]Fault

// ParsePlan reads the X-Detsim-Fault header, e.g.
// "1:die@137;2:stall@20/300ms,drip/2ms;*:chunk=3".
func ParsePlan(s string) (Plan, error) {
	p := Plan{}
	s = strings.TrimSpace(s)
	if s == "" {
		return p, nil
	}
	for _, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, spec, ok := strings.Cut(part, ":")
		if !ok {
			return nil, fmt.Errorf("fault %q: want attempt:spec", part)
		}
		n := 0
		if key != "*" {
			v, err := strconv.Atoi(key)
			if err != nil || v < 1 {
				return nil, fmt.Errorf("fault %q: bad attempt", part)
			}
			n = v
		}
		f, err := parseFault(spec)
		if err != nil {
			return nil, fmt.Errorf("fault %q: %w", part, err)
		}
		p[n] = f
	}
	return p, nil
}

func parseFault(spec string) (Fault, error) {
	var f Fault
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		switch {
		case item == "":
		case item == "holdback":
			f.Holdback = true
		case item == "5xx":
			f.Kind = "5xx"
		case strings.HasPrefix(item, "drip/"):
			d, err := time.ParseDuration(item[5:])
			if err != nil {
				return f, err
			}
			f.Drip = d
		case strings.HasPrefix(item, "slowfirst/"):
			d, err := time.ParseDuration(item[10:])
			if err != nil {
				return f, err
			}
			f.SlowFirst = d
		case strings.HasPrefix(item, "chunk="):
			n, err := strconv.Atoi(item[6:])
			if err != nil || n < 1 {
				return f, fmt.Errorf("bad chunk %q", item)
			}
			f.Chunk = n
		default:
			kind, rest, ok := strings.Cut(item, "@")
			if !ok {
				return f, fmt.Errorf("unknown fault %q", item)
			}
			switch kind {
			case "die", "cut", "eof", "stall", "err", "abort":
			default:
				return f, fmt.Errorf("unknown fault %q", kind)
			}
			at, pause, _ := strings.Cut(rest, "/")
			n, err := strconv.Atoi(at)
			if err != nil || n < 0 {
				return f, fmt.Errorf("bad position %q", at)
			}
			f.Kind, f.At = kind, n
			if kind == "stall" {
				d, err := time.ParseDuration(pause)
				if err != nil {
					return f, fmt.Errorf("stall needs a duration: %w", err)
				}
				f.Pause = d
			}
		}
	}
	return f, nil
}

func (p Plan) For(attempt int) Fault {
	if f, ok := p[attempt]; ok {
		return f
	}
	return p[0]
}

// String renders the plan in the header format, attempts in order.
func (p Plan) String() string {
	var parts []string
	for n := 0; n <= 16; n++ {
		f, ok := p[n]
		if !ok {
			continue
		}
		key := strconv.Itoa(n)
		if n == 0 {
			key = "*"
		}
		parts = append(parts, key+":"+f.String())
	}
	return strings.Join(parts, ";")
}

func (f Fault) String() string {
	var items []string
	switch f.Kind {
	case "":
	case "5xx":
		items = append(items, "5xx")
	case "stall":
		items = append(items, fmt.Sprintf("stall@%d/%s", f.At, f.Pause))
	default:
		items = append(items, fmt.Sprintf("%s@%d", f.Kind, f.At))
	}
	if f.Drip > 0 {
		items = append(items, "drip/"+f.Drip.String())
	}
	if f.SlowFirst > 0 {
		items = append(items, "slowfirst/"+f.SlowFirst.String())
	}
	if f.Holdback {
		items = append(items, "holdback")
	}
	if f.Chunk > 1 {
		items = append(items, "chunk="+strconv.Itoa(f.Chunk))
	}
	return strings.Join(items, ",")
}
