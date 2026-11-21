// Command detsim serves a deterministic fake LLM behind vLLM's
// token-in/token-out API.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/JMBugarin/parity/internal/detsim"
)

func main() {
	var (
		addr      = flag.String("addr", ":8000", "listen address")
		name      = flag.String("name", os.Getenv("HOSTNAME"), "replica name")
		model     = flag.String("model", "detsim", "served model name")
		seed      = flag.Uint64("seed", 1, "model identity; replicas of one model share it")
		eos       = flag.Uint64("eos-period", 0, "mean tokens between EOS for greedy decoding (0: never)")
		ttft      = flag.Duration("ttft-base", 0, "fixed prefill overhead")
		prefill   = flag.Duration("prefill-per-token", 0, "prefill time per uncached prompt token")
		itl       = flag.Duration("itl", 0, "decode time per token")
		jitter    = flag.Float64("itl-jitter", 0, "relative jitter on the decode time")
		perActive = flag.Float64("itl-per-active", 0, "relative decode slowdown per additional running request")
		cache     = flag.Int("prefix-cache-blocks", 0, "prefix cache capacity in 16-token blocks (0: off)")
		maxLen    = flag.Int("max-model-len", 32768, "context length")
		fixed     = flag.Bool("fixed-detok", false, "emulate a detokenizer that handles prompts ending mid-character")
		fault     = flag.String("default-fault", "", "fault plan for requests without X-Detsim-Fault")
		onTerm    = flag.String("sigterm", "drain", "on SIGTERM: drain (finish running requests) or exit")
		drainFor  = flag.Duration("drain-timeout", 25*time.Second, "longest drain before exiting")
		termDelay = flag.Duration("sigterm-delay", 0, "keep serving this long after SIGTERM before draining")
	)
	flag.Parse()
	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	s := detsim.New(detsim.Config{
		Name: *name, ModelName: *model, Seed: *seed, EOSPeriod: *eos,
		TTFTBase: *ttft, PrefillPerToken: *prefill, ITL: *itl, ITLJitter: *jitter, ITLPerActive: *perActive,
		CacheBlocks: *cache, MaxModelLen: *maxLen, FixedDetok: *fixed, DefaultFault: *fault, Logger: log,
	})
	srv := &http.Server{Addr: *addr, Handler: s, ReadHeaderTimeout: 10 * time.Second}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()
	log.Info("detsim listening", "addr", *addr, "name", *name, "seed", *seed)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	select {
	case err := <-errCh:
		if !errors.Is(err, http.ErrServerClosed) {
			log.Error("server", "err", err)
			os.Exit(1)
		}
		return
	case <-sig:
	}
	if *onTerm == "exit" {
		log.Info("exiting on SIGTERM without draining", "running", s.Stats.Running.Load())
		os.Exit(0)
	}
	time.Sleep(*termDelay)
	s.Drain()
	log.Info("draining", "running", s.Stats.Running.Load())
	ctx, cancel := context.WithTimeout(context.Background(), *drainFor)
	defer cancel()
	_ = s.WaitIdle(ctx)
	_ = srv.Shutdown(ctx)
	log.Info("stopped", "running", s.Stats.Running.Load())
}
