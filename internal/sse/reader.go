// Package sse reads and writes Server-Sent Events as specified by the WHATWG
// HTML standard (section 9.2), with a size limit and explicit truncation
// errors so callers can tell a clean end of stream from a cut-off event.
package sse

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strconv"
)

// Event is one dispatched SSE event, or one comment line when the reader
// was created with ReturnComments.
type Event struct {
	Type      string
	Data      []byte
	ID        string
	HasID     bool
	Retry     int
	IsComment bool
}

var (
	// ErrTruncated means the stream ended after some lines of an event
	// but before the blank line that dispatches it. The partial event is
	// discarded, as the standard requires.
	ErrTruncated = errors.New("sse: stream ended inside an event")
	// ErrTooLarge means a single event exceeded the reader's size limit.
	ErrTooLarge = errors.New("sse: event exceeds size limit")
)

const DefaultMaxEventSize = 1 << 20

var bom = []byte{0xEF, 0xBB, 0xBF}

// Reader parses an event stream. It is not safe for concurrent use.
type Reader struct {
	br             *bufio.Reader
	max            int
	ReturnComments bool

	started bool
	skipLF  bool
	line    []byte

	ev      Event
	data    []byte
	hasData bool
	inEvent bool
	size    int
}

func NewReader(r io.Reader, maxEventSize int) *Reader {
	if maxEventSize <= 0 {
		maxEventSize = DefaultMaxEventSize
	}
	rd := &Reader{br: bufio.NewReaderSize(r, 16<<10), max: maxEventSize}
	rd.reset()
	return rd
}

func (r *Reader) reset() {
	r.ev = Event{Retry: -1}
	r.data = r.data[:0]
	r.hasData = false
	r.inEvent = false
	r.size = 0
}

var errPartialLine = errors.New("partial line")

func (r *Reader) readLine() ([]byte, error) {
	r.line = r.line[:0]
	for {
		b, err := r.br.ReadByte()
		if err != nil {
			if err == io.EOF && len(r.line) > 0 {
				return nil, errPartialLine
			}
			return nil, err
		}
		if r.skipLF {
			r.skipLF = false
			if b == '\n' {
				continue
			}
		}
		switch b {
		case '\n':
			return r.line, nil
		case '\r':
			r.skipLF = true
			return r.line, nil
		}
		if len(r.line) >= r.max {
			return nil, ErrTooLarge
		}
		r.line = append(r.line, b)
	}
}

// Next returns the next event. It returns io.EOF when the stream ends on an
// event boundary and ErrTruncated when it ends inside one.
func (r *Reader) Next() (Event, error) {
	for {
		line, err := r.readLine()
		if err != nil {
			switch {
			case err == errPartialLine:
				return Event{}, ErrTruncated
			case err == io.EOF && r.inEvent:
				return Event{}, ErrTruncated
			}
			return Event{}, err
		}
		if !r.started {
			r.started = true
			line = bytes.TrimPrefix(line, bom)
		}
		if len(line) == 0 {
			if !r.hasData {
				r.reset()
				continue
			}
			ev := r.ev
			ev.Data = append([]byte(nil), r.data...)
			r.reset()
			return ev, nil
		}
		r.size += len(line) + 1
		if r.size > r.max {
			return Event{}, ErrTooLarge
		}
		if line[0] == ':' {
			if r.ReturnComments {
				c := line[1:]
				if len(c) > 0 && c[0] == ' ' {
					c = c[1:]
				}
				return Event{IsComment: true, Data: append([]byte(nil), c...), Retry: -1}, nil
			}
			continue
		}
		r.inEvent = true
		field, value := line, []byte(nil)
		if i := bytes.IndexByte(line, ':'); i >= 0 {
			field, value = line[:i], line[i+1:]
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
		}
		switch string(field) {
		case "event":
			r.ev.Type = string(value)
		case "data":
			if r.hasData {
				r.data = append(r.data, '\n')
			}
			r.data = append(r.data, value...)
			r.hasData = true
		case "id":
			if bytes.IndexByte(value, 0) < 0 {
				r.ev.ID = string(value)
				r.ev.HasID = true
			}
		case "retry":
			if isDigits(value) {
				if n, err := strconv.Atoi(string(value)); err == nil {
					r.ev.Retry = n
				}
			}
		}
	}
}

func isDigits(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	for _, c := range b {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
