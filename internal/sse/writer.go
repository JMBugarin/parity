package sse

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
)

// Writer frames events onto an io.Writer and flushes after each one when
// the destination supports it.
type Writer struct {
	w   io.Writer
	f   http.Flusher
	buf bytes.Buffer
}

func NewWriter(w io.Writer) *Writer {
	f, _ := w.(http.Flusher)
	return &Writer{w: w, f: f}
}

// Encode appends the wire form of ev to dst. CR and CRLF inside Data are
// normalized to LF because a bare CR would end the line on the reading side.
func Encode(dst []byte, ev Event) []byte {
	if ev.IsComment {
		for _, line := range splitLines(ev.Data) {
			dst = append(dst, ':', ' ')
			dst = append(dst, line...)
			dst = append(dst, '\n')
		}
		return append(dst, '\n')
	}
	if ev.Type != "" {
		dst = append(dst, "event: "...)
		dst = append(dst, ev.Type...)
		dst = append(dst, '\n')
	}
	if ev.HasID {
		dst = append(dst, "id: "...)
		dst = append(dst, ev.ID...)
		dst = append(dst, '\n')
	}
	if ev.Retry >= 0 {
		dst = append(dst, "retry: "...)
		dst = strconv.AppendInt(dst, int64(ev.Retry), 10)
		dst = append(dst, '\n')
	}
	for _, line := range splitLines(ev.Data) {
		dst = append(dst, "data: "...)
		dst = append(dst, line...)
		dst = append(dst, '\n')
	}
	return append(dst, '\n')
}

func splitLines(b []byte) [][]byte {
	if bytes.IndexByte(b, '\r') >= 0 {
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		b = bytes.ReplaceAll(b, []byte("\r"), []byte("\n"))
	}
	return bytes.Split(b, []byte("\n"))
}

func (w *Writer) Write(ev Event) error {
	w.buf.Reset()
	w.buf.Write(Encode(w.buf.AvailableBuffer(), ev))
	if _, err := w.w.Write(w.buf.Bytes()); err != nil {
		return err
	}
	if w.f != nil {
		w.f.Flush()
	}
	return nil
}

// Data writes one unnamed event carrying data.
func (w *Writer) Data(data []byte) error {
	return w.Write(Event{Data: data, Retry: -1})
}

func (w *Writer) Comment(text string) error {
	return w.Write(Event{IsComment: true, Data: []byte(text), Retry: -1})
}
