package detsim

// Model is the deterministic "network": next token = H(seed, whole prefix).
type Model struct {
	Vocab *Vocab
	Seed  uint64
	// EOSPeriod makes greedy decoding stop with probability 1/EOSPeriod per
	// token unless ignore_eos is set. Zero disables EOS.
	EOSPeriod uint64
}

func splitmix(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// State is the running hash of a token prefix.
type State struct {
	h    uint64
	last int
	n    int
}

func (m *Model) Start() State { return State{h: splitmix(m.Seed), last: -1} }

func (s State) Push(tok int) State {
	return State{h: splitmix(s.h ^ (uint64(tok)+1)*0xD6E8FEB86659FD93), last: tok, n: s.n + 1}
}

func (m *Model) Prefix(ids []int) State {
	s := m.Start()
	for _, t := range ids {
		s = s.Push(t)
	}
	return s
}

// Sampling holds the parameters detsim honors.
type Sampling struct {
	Temperature float64
	Seed        uint64
	HasSeed     bool
	IgnoreEOS   bool
	StopTokens  map[int]bool
}

// Next returns the next token after the prefix summarized by s. step is the
// index of this token within the current request: a sampled request draws
// from an RNG that restarts at zero on every request, so a resumed sampled
// stream continues validly but not identically, as a real engine would.
func (m *Model) Next(s State, step int, sp Sampling) int {
	v := m.Vocab
	if s.last >= 0 && s.last < len(v.Tokens) {
		if nx := v.Tokens[s.last].Next; nx >= 0 {
			return nx
		}
	}
	h := s.h
	if sp.Temperature > 0 {
		h = splitmix(h ^ splitmix(sp.Seed+uint64(step)*0x2545F4914F6CDD1D))
	}
	if m.EOSPeriod > 0 && !sp.IgnoreEOS && (h>>40)%m.EOSPeriod == 0 {
		return TokIMEnd
	}
	return v.gen[h%uint64(len(v.gen))]
}

// Run generates up to max tokens greedily or by sampling, starting after
// prompt. It returns the tokens and whether generation stopped on EOS or a
// stop token.
func (m *Model) Run(prompt []int, max int, sp Sampling) ([]int, bool) {
	s := m.Prefix(prompt)
	var out []int
	for i := 0; i < max; i++ {
		t := m.Next(s, i, sp)
		out = append(out, t)
		if t == TokIMEnd || sp.StopTokens[t] {
			return out, true
		}
		s = s.Push(t)
	}
	return out, false
}
