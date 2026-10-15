GO ?= go

.PHONY: all build test fuzz tidy fmt vet check clean

## all: default target — build everything
all: build

## build: compile all packages and the needle-ask CLI
## (linking cmd/needle-ask needs the static libc++/libc++abi that
## engine/cgo/libs were built against; see README for the matrix)
build:
	$(GO) build ./...
	$(GO) build -o needle-ask ./cmd/needle-ask

## test: run the unit tests with the race detector. The fuzz differential
## (fuzzdiff_test.go) is skipped — it only runs via the fuzz target.
test:
	$(GO) test -race -count=1 -timeout 30m ./...

## fuzz: run the differential fuzzer (TestFuzzDifferential) against the
## cgo engine. Requires cgo + linux and the static libc++/libc++abi
## runtime; ~100 s without -race. NEEDLE_FUZZDIFF=1 opts the test in.
fuzz:
	NEEDLE_FUZZDIFF=1 $(GO) test -count=1 -run TestFuzzDifferential -timeout 60m .

## tidy: verify go.mod/go.sum are tidy and committed
tidy:
	$(GO) mod tidy
	git diff --exit-code -- go.mod go.sum

## fmt: fail if any file is not gofmt-formatted
fmt:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then \
	  echo "The following files are not gofmt-formatted:"; \
	  echo "$$unformatted"; \
	  exit 1; \
	fi

## vet: run go vet over all packages
vet:
	$(GO) vet ./...

## check: everything CI runs, in the same order
check: tidy build fmt vet test

## clean: remove build artifacts
clean:
	$(GO) clean
	rm -f needle-ask
