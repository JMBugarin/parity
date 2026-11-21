package detsim

import "strings"

// DecodeStream reproduces the incremental detokenizer vLLM v0.34.0 uses for
// fast tokenizers (tokenizers.decoders.DecodeStream primed with the prompt
// in one step). A step emits text only when the decoded string grows and
// does not end in U+FFFD. Because the prompt is one step, a prompt that ends
// inside a multi-byte character never sets the prefix, and the first
// emission after it repeats the whole prompt. That is the behavior measured
// against vLLM in docs/evidence/M0; Fixed disables it.
type DecodeStream struct {
	v           *Vocab
	skipSpecial bool
	ids         []int
	prefix      string
	prefixIndex int
	Fixed       bool
}

func NewDecodeStream(v *Vocab, prompt []int, skipSpecial, fixed bool) *DecodeStream {
	d := &DecodeStream{v: v, skipSpecial: skipSpecial, Fixed: fixed}
	if fixed {
		// Prime only up to the last clean boundary, so held bytes from the
		// prompt carry into the first output token.
		clean := len(prompt)
		for clean > 0 && strings.HasSuffix(v.Decode(prompt[:clean], skipSpecial), "\uFFFD") {
			clean--
		}
		d.ids = append(d.ids, prompt[:clean]...)
		d.prefix = v.Decode(d.ids, skipSpecial)
		d.prefixIndex = 0
		d.ids = append(d.ids, prompt[clean:]...)
		return d
	}
	d.step(prompt)
	return d
}

func (d *DecodeStream) step(ids []int) (string, bool) {
	d.ids = append(d.ids, ids...)
	s := d.v.Decode(d.ids, d.skipSpecial)
	if len(s) > len(d.prefix) && !strings.HasSuffix(s, "\uFFFD") {
		if !strings.HasPrefix(s, d.prefix) {
			// tokenizers raises InvalidPrefix here and vLLM resets the
			// stream with no prompt context.
			d.ids = append(d.ids[:0], ids...)
			d.prefix, d.prefixIndex = "", 0
			return d.step(nil)
		}
		out := s[len(d.prefix):]
		newIndex := len(d.ids) - d.prefixIndex
		d.ids = append([]int(nil), d.ids[d.prefixIndex:]...)
		d.prefix = d.v.Decode(d.ids, d.skipSpecial)
		d.prefixIndex = newIndex
		return out, true
	}
	return "", false
}

// Step feeds one generated token and returns the text it releases.
func (d *DecodeStream) Step(tok int) string {
	s, _ := d.step([]int{tok})
	return s
}
