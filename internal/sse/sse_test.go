package sse

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"testing/iotest"
)

func readAll(t *testing.T, r io.Reader) ([]Event, error) {
	t.Helper()
	rd := NewReader(r, 0)
	var out []Event
	for {
		ev, err := rd.Next()
		if err != nil {
			if err == io.EOF {
				return out, nil
			}
			return out, err
		}
		out = append(out, ev)
	}
}

func TestReaderSpecCases(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
		err  error
	}{
		{"lf", "data: a\n\ndata: b\n\n", []string{"a", "b"}, nil},
		{"crlf", "data: a\r\n\r\ndata: b\r\n\r\n", []string{"a", "b"}, nil},
		{"cr", "data: a\r\rdata: b\r\r", []string{"a", "b"}, nil},
		{"multiline", "data: a\ndata: b\n\n", []string{"a\nb"}, nil},
		{"no space", "data:a\n\n", []string{"a"}, nil},
		{"two spaces", "data:  a\n\n", []string{" a"}, nil},
		{"empty data dispatches", "data\n\n", []string{""}, nil},
		{"no data no dispatch", "event: x\n\ndata: y\n\n", []string{"y"}, nil},
		{"comments skipped", ": hi\ndata: a\n: there\n\n", []string{"a"}, nil},
		{"bom", "\xEF\xBB\xBFdata: a\n\n", []string{"a"}, nil},
		{"unknown field", "foo: bar\ndata: a\n\n", []string{"a"}, nil},
		{"truncated", "data: a\n\ndata: b\n", []string{"a"}, ErrTruncated},
		{"truncated mid line", "data: a\n\ndata: b", []string{"a"}, ErrTruncated},
		{"trailing comment ok", "data: a\n\n: keepalive\n", []string{"a"}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			evs, err := readAll(t, strings.NewReader(c.in))
			if !errors.Is(err, c.err) {
				t.Fatalf("err = %v, want %v", err, c.err)
			}
			var got []string
			for _, ev := range evs {
				got = append(got, string(ev.Data))
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Fatalf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestReaderFieldsAndComments(t *testing.T) {
	rd := NewReader(strings.NewReader("event: e\nid: 7\nretry: 300\n: note\ndata: x\n\n"), 0)
	rd.ReturnComments = true
	c, err := rd.Next()
	if err != nil || !c.IsComment || string(c.Data) != "note" {
		t.Fatalf("comment = %+v, %v", c, err)
	}
	ev, err := rd.Next()
	if err != nil {
		t.Fatal(err)
	}
	if ev.Type != "e" || ev.ID != "7" || !ev.HasID || ev.Retry != 300 || string(ev.Data) != "x" {
		t.Fatalf("event = %+v", ev)
	}
}

func TestReaderOneByteAtATime(t *testing.T) {
	in := "data: {\"a\":1}\r\n\r\ndata: b\rdata: c\r\r"
	evs, err := readAll(t, iotest.OneByteReader(strings.NewReader(in)))
	if err != nil {
		t.Fatal(err)
	}
	if len(evs) != 2 || string(evs[0].Data) != `{"a":1}` || string(evs[1].Data) != "b\nc" {
		t.Fatalf("got %+v", evs)
	}
}

func TestReaderSizeLimit(t *testing.T) {
	rd := NewReader(strings.NewReader("data: "+strings.Repeat("x", 100)+"\n\n"), 64)
	if _, err := rd.Next(); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v", err)
	}
	rd = NewReader(strings.NewReader(strings.Repeat("data: xxxxxxxx\n", 20)+"\n"), 64)
	if _, err := rd.Next(); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("multi-line err = %v", err)
	}
}

func TestReaderDispatchesWithoutLookahead(t *testing.T) {
	pr, pw := io.Pipe()
	defer pr.Close()
	go func() {
		pw.Write([]byte("data: a\r\r"))
	}()
	ev, err := NewReader(pr, 0).Next()
	if err != nil || string(ev.Data) != "a" {
		t.Fatalf("got %+v %v", ev, err)
	}
	pw.Close()
}

func TestRoundTrip(t *testing.T) {
	evs := []Event{
		{Data: []byte(`{"x":1}`), Retry: -1},
		{Type: "t", ID: "9", HasID: true, Retry: 5, Data: []byte("a\nb")},
		{Data: []byte(""), Retry: -1},
		{Data: []byte("[DONE]"), Retry: -1},
	}
	var buf []byte
	for _, ev := range evs {
		buf = Encode(buf, ev)
	}
	got, err := readAll(t, bytes.NewReader(buf))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(evs) {
		t.Fatalf("got %d events, want %d", len(got), len(evs))
	}
	for i := range evs {
		g, w := got[i], evs[i]
		if !bytes.Equal(g.Data, w.Data) || g.Type != w.Type || g.ID != w.ID || g.HasID != w.HasID || g.Retry != w.Retry {
			t.Fatalf("event %d: got %+v want %+v", i, g, w)
		}
	}
}

// TestRecordedStreamsByteExact re-frames every recorded upstream stream event
// by event and requires the output to match the recording byte for byte.
func TestRecordedStreamsByteExact(t *testing.T) {
	files, _ := filepath.Glob("testdata/*.sse")
	if len(files) == 0 {
		t.Skip("no recordings")
	}
	for _, f := range files {
		t.Run(filepath.Base(f), func(t *testing.T) {
			raw, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			evs, err := readAll(t, bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			if len(evs) < 2 {
				t.Fatalf("only %d events", len(evs))
			}
			var out bytes.Buffer
			w := NewWriter(&out)
			for _, ev := range evs {
				if err := w.Write(ev); err != nil {
					t.Fatal(err)
				}
			}
			if !bytes.Equal(out.Bytes(), raw) {
				t.Fatalf("re-framed stream differs from recording (%d vs %d bytes)", out.Len(), len(raw))
			}
		})
	}
}

func FuzzReader(f *testing.F) {
	f.Add([]byte("data: a\n\n"))
	f.Add([]byte("data: a\r\ndata: b\r\n\r\n: c\nevent: x\nid: 1\nretry: 2\n\n"))
	f.Add([]byte("\xEF\xBB\xBFdata\n\ndata: [DONE]\n\n"))
	f.Add([]byte("data: {\"choices\":[{\"text\":\"\\u00e9\"}]}\n\ndata: tr"))
	f.Fuzz(func(t *testing.T, in []byte) {
		rd := NewReader(bytes.NewReader(in), 4096)
		var evs []Event
		for {
			ev, err := rd.Next()
			if err != nil {
				break
			}
			if len(ev.Data) > 4096 {
				t.Fatalf("event larger than limit: %d", len(ev.Data))
			}
			evs = append(evs, ev)
		}
		var buf []byte
		for _, ev := range evs {
			buf = Encode(buf, ev)
		}
		again, err := readAll(t, bytes.NewReader(buf))
		if err != nil {
			t.Fatalf("re-read of encoded events failed: %v", err)
		}
		if len(again) != len(evs) {
			t.Fatalf("round trip changed event count %d -> %d", len(evs), len(again))
		}
		for i := range evs {
			want := bytes.ReplaceAll(bytes.ReplaceAll(evs[i].Data, []byte("\r\n"), []byte("\n")), []byte("\r"), []byte("\n"))
			if !bytes.Equal(again[i].Data, want) || again[i].Type != evs[i].Type {
				t.Fatalf("round trip changed event %d: %q -> %q", i, evs[i].Data, again[i].Data)
			}
		}
	})
}
