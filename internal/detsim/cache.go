package detsim

import (
	"container/list"
	"sync"
)

const BlockSize = 16

// PrefixCache models an engine's prefix cache as an LRU of chained block
// hashes, so a resume on a replica that already holds the prefix prefills
// faster than one that does not.
type PrefixCache struct {
	mu    sync.Mutex
	cap   int
	ll    *list.List
	items map[uint64]*list.Element
}

func NewPrefixCache(blocks int) *PrefixCache {
	return &PrefixCache{cap: blocks, ll: list.New(), items: map[uint64]*list.Element{}}
}

func blockHashes(ids []int) []uint64 {
	n := len(ids) / BlockSize
	out := make([]uint64, n)
	h := uint64(0xCBF29CE484222325)
	for b := 0; b < n; b++ {
		for _, t := range ids[b*BlockSize : (b+1)*BlockSize] {
			h = splitmix(h ^ uint64(t+1))
		}
		out[b] = h
	}
	return out
}

// Match returns how many leading tokens of ids are cached.
func (c *PrefixCache) Match(ids []int) int {
	if c == nil || c.cap == 0 {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, h := range blockHashes(ids) {
		e, ok := c.items[h]
		if !ok {
			break
		}
		c.ll.MoveToFront(e)
		n += BlockSize
	}
	return n
}

// Insert records every full block of ids.
func (c *PrefixCache) Insert(ids []int) {
	if c == nil || c.cap == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, h := range blockHashes(ids) {
		if e, ok := c.items[h]; ok {
			c.ll.MoveToFront(e)
			continue
		}
		c.items[h] = c.ll.PushFront(h)
		for c.ll.Len() > c.cap {
			old := c.ll.Back()
			c.ll.Remove(old)
			delete(c.items, old.Value.(uint64))
		}
	}
}
