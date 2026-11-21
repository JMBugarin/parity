GO ?= go
BIN := bin

.PHONY: all build test race fuzz vet fmt check
all: check build

build:
	$(GO) build -o $(BIN)/ ./cmd/...

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

fuzz:
	$(GO) test ./internal/sse -run '^$$' -fuzz FuzzReader -fuzztime 20s

vet:
	$(GO) vet ./...

fmt:
	gofmt -w cmd internal

check:
	test -z "$$(gofmt -l .)"
	$(GO) vet ./...
	$(GO) test -race ./...
