// Package detsim is a deterministic fake LLM engine that serves vLLM's
// token-in/token-out API. The next token is a hash of a model seed and the
// whole token prefix, so any continuation is reproducible from the prefix
// alone: an uninterrupted run and a correctly spliced run produce the same
// tokens, and any lost, duplicated or reordered token changes everything
// after it.
package detsim

import (
	"strings"
	"unicode/utf8"
)

// Token IDs with fixed meaning.
const (
	TokPad     = 0
	TokIMStart = 1
	TokIMEnd   = 2
	byteBase   = 3
)

// Token is one vocabulary entry.
type Token struct {
	Bytes   []byte
	Special bool
	// Next forces the following token, which models a multi-byte character
	// that the tokenizer splits across several tokens.
	Next int
	// Start marks tokens the model may choose freely.
	Start bool
}

// Vocab is detsim's tokenizer. Byte fallback tokens make any text
// encodable; a handful of tokens split multi-byte UTF-8 characters, and a few
// short pieces make stop strings straddle token boundaries.
type Vocab struct {
	Tokens []Token
	gen    []int
	trie   *trieNode
}

type trieNode struct {
	next map[byte]*trieNode
	id   int
}

var words = []string{
	" the", " quick", " brown", " fox", " jumps", " over", " lazy", " dog", " and",
	" then", " startled", " runs", " a", " of", " to", " in", " is", " it", " that",
	" was", " for", " on", " are", " with", " as", " his", " they", " be", " at",
	" one", " have", " this", " from", " by", " hot", " word", " but", " what",
	" some", " we", " can", " out", " other", " were", " all", " there", " when",
	" up", " use", " your", " how", " said", " an", " each", " she", " which", " do",
	" their", " time", " if", " will", " way", " about", " many", " write", " would",
	" like", " so", " these", " her", " long", " make", " thing", " see", " him",
	" two", " has", " look", " more", " day", " could", " go", " come", " did",
	" number", " sound", " no", " most", " people", " my", " water", " than", " call",
	" first", " who", " may", " down", " side", " been", " now", " find", " any",
	" new", " work", " part", " take", " get", " place", " made", " live", " where",
	" after", " back", " little", " only", " round", " man", " year", " came",
	" show", " every", " good", " me", " give", " our", " under", " name", " very",
	" through", " just", " form", " sentence", " great", " think", " say", " help",
	" low", " line", " differ", " turn", " cause", " much", " mean", " before",
	" move", " right", " boy", " old", " too", " same", " tell", " does", " set",
	" three", " want", " air", " well", " also", " play", " small", " end", " put",
	" home", " read", " hand", " port", " large", " spell", " add", " even", " land",
	" here", " must", " big", " high", " such", " follow", " act", " why", " ask",
	" men", " change", " went", " light", " kind", " off", " need", " house",
	" picture", " try", " us", " again", " animal", " point", " mother", " world",
	" near", " build", " self", " earth", " father", " head", " stand", " own",
	" page", " should", " country", " found", " answer", " school", " grow",
	" study", " still", " learn", " plant", " cover", " food", " sun", " four",
	" between", " state", " keep", " eye", " never", " last", " let", " thought",
	" city", " tree", " cross", " farm", " hard", " start", " might", " story",
	" saw", " far", " sea", " draw", " left", " late", " run", " while", " press",
	" close", " night", " real", " life", " few", " north", ".", ",", ";", ":", "!",
	"?", "\n", " 0", " 1", " 2", " 3", " 42", " 137", "The", "And", "But", "So",
	"é", "ü", "日本", "語", " 🦊", "—", "…", "ST", "OP", "STOP", " ST", "<|", "end",
	"|>", "###", "END", " E", "ND", "\n\n",
}

// chains are characters the vocabulary only has as several tokens, listed
// in order. The first piece can be chosen; the rest are forced.
var chains = [][]string{
	{" \xF0\x9F", "\xA6", "\x8A"},            // " 🦊" as three tokens, as Qwen3 splits it
	{"\xE6\x97", "\xA5"},                     // "日"
	{"\xE6\x97", "\xA5\xE6\x9C", "\xAC"},     // "日本", middle piece straddles two characters
	{"\xC3", "\xA9"},                         // "é"
	{"\xE2\x80", "\xA6"},                     // "…"
	{"\xF0\x9F\x98", "\x80"},                 // "😀"
	{" caf", "\xC3", "\xA9"},                 // " café"
	{"\xE8\xAA", "\x9E", "\xE6\x97\xA5"},     // "語日"
	{"\xF0", "\x9F\x8C", "\x8D"},             // "🌍"
	{"\xF0\x9F\x8C\x8D\xF0\x9F", "\xA6\x8A"}, // "🌍🦊"
}

// NewVocab builds the fixed detsim vocabulary.
func NewVocab() *Vocab {
	v := &Vocab{trie: &trieNode{id: -1}}
	add := func(t Token) int {
		v.Tokens = append(v.Tokens, t)
		return len(v.Tokens) - 1
	}
	add(Token{Bytes: []byte("<|endoftext|>"), Special: true})
	add(Token{Bytes: []byte("<|im_start|>"), Special: true})
	add(Token{Bytes: []byte("<|im_end|>"), Special: true})
	for b := 0; b < 256; b++ {
		add(Token{Bytes: []byte{byte(b)}, Next: -1})
	}
	for _, w := range words {
		v.gen = append(v.gen, add(Token{Bytes: []byte(w), Next: -1, Start: true}))
	}
	for _, c := range chains {
		ids := make([]int, len(c))
		for i, piece := range c {
			ids[i] = add(Token{Bytes: []byte(piece), Next: -1, Start: i == 0})
		}
		for i := 0; i+1 < len(ids); i++ {
			v.Tokens[ids[i]].Next = ids[i+1]
		}
		v.gen = append(v.gen, ids[0])
	}
	for id, t := range v.Tokens {
		if t.Special || len(t.Bytes) == 0 {
			continue
		}
		v.insert(t.Bytes, id)
	}
	return v
}

func (v *Vocab) insert(b []byte, id int) {
	n := v.trie
	for _, c := range b {
		if n.next == nil {
			n.next = map[byte]*trieNode{}
		}
		child := n.next[c]
		if child == nil {
			child = &trieNode{id: -1}
			n.next[c] = child
		}
		n = child
	}
	// Keep the first ID for a byte string so encoding stays stable.
	if n.id < 0 {
		n.id = id
	}
}

func (v *Vocab) Size() int { return len(v.Tokens) }

// Encode tokenizes text by greedy longest match, falling back to bytes.
func (v *Vocab) Encode(s string) []int {
	b := []byte(s)
	var out []int
	for i := 0; i < len(b); {
		n, best, bestLen := v.trie, -1, 0
		for j := i; j < len(b); j++ {
			n = n.next[b[j]]
			if n == nil {
				break
			}
			if n.id >= 0 {
				best, bestLen = n.id, j-i+1
			}
		}
		if best < 0 {
			best, bestLen = byteBase+int(b[i]), 1
		}
		out = append(out, best)
		i += bestLen
	}
	return out
}

// Bytes concatenates token bytes, optionally skipping special tokens.
func (v *Vocab) Bytes(ids []int, skipSpecial bool) []byte {
	var out []byte
	for _, id := range ids {
		if id < 0 || id >= len(v.Tokens) {
			continue
		}
		t := v.Tokens[id]
		if t.Special && skipSpecial {
			continue
		}
		out = append(out, t.Bytes...)
	}
	return out
}

// Decode is the tokenizer's batch decode: UTF-8 with each maximal invalid
// subpart replaced by U+FFFD, like Rust's String::from_utf8_lossy.
func (v *Vocab) Decode(ids []int, skipSpecial bool) string {
	return Lossy(v.Bytes(ids, skipSpecial))
}

// ChatPrompt applies detsim's chat template, a ChatML subset.
func (v *Vocab) ChatPrompt(msgs []Message) []int {
	var ids []int
	for _, m := range msgs {
		ids = append(ids, TokIMStart)
		ids = append(ids, v.Encode(m.Role+"\n"+m.Content)...)
		ids = append(ids, TokIMEnd)
		ids = append(ids, v.Encode("\n")...)
	}
	ids = append(ids, TokIMStart)
	ids = append(ids, v.Encode("assistant\n")...)
	return ids
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Lossy implements the maximal-subpart replacement of invalid UTF-8.
func Lossy(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var sb strings.Builder
	for i := 0; i < len(b); {
		r, size := utf8.DecodeRune(b[i:])
		if r != utf8.RuneError || size > 1 {
			sb.Write(b[i : i+size])
			i += size
			continue
		}
		sb.WriteRune(utf8.RuneError)
		i += maximalSubpart(b[i:])
	}
	return sb.String()
}

func maximalSubpart(b []byte) int {
	lead := b[0]
	var need int
	lo, hi := byte(0x80), byte(0xBF)
	switch {
	case lead >= 0xC2 && lead <= 0xDF:
		need = 1
	case lead == 0xE0:
		need, lo = 2, 0xA0
	case lead >= 0xE1 && lead <= 0xEC, lead == 0xEE, lead == 0xEF:
		need = 2
	case lead == 0xED:
		need, hi = 2, 0x9F
	case lead == 0xF0:
		need, lo = 3, 0x90
	case lead >= 0xF1 && lead <= 0xF3:
		need = 3
	case lead == 0xF4:
		need, hi = 3, 0x8F
	default:
		return 1
	}
	n := 1
	for k := 0; k < need && n < len(b); k++ {
		c := b[n]
		if c < lo || c > hi {
			break
		}
		lo, hi = 0x80, 0xBF
		n++
	}
	return n
}
