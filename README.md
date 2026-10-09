# needle-go

Go library + example CLIs for [Needle 3](https://github.com/cactus-compute/needle)
([weights](https://huggingface.co/Cactus-Compute/needle3)) — Cactus Compute's
tiny on-device tool-calling model. The API mirrors the Python `cactus-needle`
package in idiomatic Go, and the demo binaries ship the **entire model inside
the executable** (`go:embed`).

Two interchangeable backends, both implementing the `needle.Engine` interface
(`Complete` / `Run` / `Reset` / `Embed`):

| | `engine/cgo` | `engine/pure` |
|---|---|---|
| What | FFI bindings to the prebuilt `libneedle.a` shipped in the HF repo (same engine as the Python wheel) | full runtime ported to Go: `.cact` CQ dequantizer, tokenizer, incremental attention, mHC lanes, engrams, Monarch-Hadamard MLP, constrained greedy decoding |
| cgo | yes (static lib per platform) | **no** — zero non-Go dependencies |
| Toolchain | C compiler **plus a libc++ runtime** (the static libs are clang/libc++ builds; ~50 `std::__1` symbols) | plain Go; not even a C toolchain required |
| Demo runtime (this machine, 8 cores) | 0.9 s full demo | ~12 s full demo (~3 s per `Complete`) |
| Weights | `.cact` read in place (~0.1 s to first token) | dequantized to FP32 at `New` (~1.5 s, ~1.1 GB RSS) |

## Quick start

```sh
go run ./cmd/needle-pure   # pure-Go engine, no cgo
go run ./cmd/needle-cgo    # prebuilt engine (linux/windows, x86_64/arm64)
```

Both CLIs run the same scripted tool-call demos (weather + smart-home tools)
and an embedding query. One binary, one embedded `needle3.cact` (33.7 MB).

## Usage

```go
import (
	needle "github.com/Zetaphor/needle-go"
	"github.com/Zetaphor/needle-go/engine/pure"
	nmodel "github.com/Zetaphor/needle-go/model"
)

agent, err := pure.New(needle.Options{
	Weights: nmodel.Needle3, // embedded .cact bytes
	Tools: []needle.Tool{{
		Name:        "get_weather",
		Description: "Current weather in a city.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		},
	}},
})
defer agent.Close()

resp, err := agent.Run("what's it like in Lagos right now?")
fmt.Println(resp.Type, resp.FunctionCalls, resp.Confidence)
```

Python ↔ Go cheat sheet (also in [needle.go](needle.go)):

| Python (`cactus-needle`) | Go |
|---|---|
| `needle.Needle(tools=[...])` | `pure.New(needle.Options{Tools: ...})` (or `cgo.New`) |
| `agent.complete(text)` | `agent.Complete(text)` |
| `agent.run(query)` | `agent.Run(query)` — agentic loop with `Tool.Fn` handlers |
| `agent.reset()` | `agent.Reset()` |
| `agent.embed(text)` | `agent.Embed(text)` → 3072-float unit vector |
| `res["function_calls"]` | `resp.FunctionCalls` |
| `res["confidence"]` | `resp.Confidence` |

## Platforms

The four platforms the upstream project ships prebuilt engines for —
`linux-x86_64`, `linux-arm64`, `windows-x86_64`, `windows-arm64` — are covered
by `engine/cgo` (static libs vendored under `engine/cgo/libs/<triple>/`).

`engine/pure` compiles anywhere Go does: cross-compile checks pass for
`windows/amd64`, `linux/arm64`, `darwin/arm64` with `CGO_ENABLED=0`.

> [!NOTE]
> "Only a C toolchain" holds for the cgo binding only in the sense that no
> C++ *compiler* is invoked — but linking `libneedle.a` needs libc++ (and on
> Linux a `libstdc++`-only link fails on `std::__1` symbols). This is why the
> pure-Go variant exists, and why the module ships both: the pure engine has
> no such requirement at all.

## Equivalence evidence

Verified against the upstream JAX reference (`needle.model`) and the C engine
on this machine:

- **CQ dequantizer**: FP16 + 4-bit codebook/Walsh–Hadamard records reproduce
  the JAX tensors exactly (golden test).
- **Forward pass**: all 21 probe cells at the final position match the JAX
  oracle within ~1e-5 relative; the top-5 first-token logits are identical,
  in order.
- **Engines agree end-to-end**: on the shared demo scripts the pure port
  produces the same calls and — for the Lagos/off-topic cases — reasoning text
  identical to the C engine, word for word.
- **Embeddings**: cosine between `pure.Embed` and `cgo.Embed` for the same
  text is 0.9999 (`embedcmp_test.go`).
- **Confidence**: geometric mean of greedy token probabilities, combined with
  the calibrated probe-head sigmoid — tracks the engine (e.g. 0.528 vs 0.529
  on the Lagos query).

## Known deviations (pure engine)

- Constrained decoding covers the JSON tool-call region (structure, schema
  keys, tool names, `[]`/`</tool_call>`/EOS ordering); the reasoning phase is
  free, so its wording can drift where the engines' near-tie logits disagree
  (e.g. "brighten" → `brightness:80` alone vs also `on:true` at conf 0.11).
- Everything runs in FP32; the C engine quantizes KV/attention internally, so
  occasional token-level divergence is expected under identical prompts.
- The engine's process-global single-model constraint is preserved: one live
  agent per kind per process (backends guard with a mutex; the pure engine is
  internally parallel over matvecs).

## Fuzz differential results

`fuzzdiff_test.go` (build `cgo && linux`, ~100 s) runs a seeded 35-prompt
corpus — 6 weather templates, 5 smart-home commands, 8 gibberish strings,
6 edge cases, 5 unicode/numeric, 5 long (~180-token) — through both engines
in one process and compares type, parsed tool calls, and reasoning per
prompt. The C ABI exposes no quantization knob, so the reference side runs
the engine's native quantized mode; the full per-case report is checked in
at [docs/fuzzdiff-report.txt](docs/fuzzdiff-report.txt).

| Metric | Pre-gate | Post-gate |
|---|---|---|
| Type agreement | 34/35 | 34/35 |
| Tool-call agreement | 14/35 | **31/35** |
| Fully exact (type + calls + reasoning) | 9/35 | **19/35** |
| Confidence delta | mean 0.041 / max 0.52 | unchanged |

What the fuzzing found:

1. **The engine suppresses low-confidence calls** — pure was missing this
   gate. Ported (`SuppressedCalls`, threshold 0.1; corpus: suppressed at
   conf ≤ 0.093, emitted at ≥ 0.116 in every case), fixing 17 of 21 call
   mismatches.
2. **In-distribution behavior agrees fully**: on all 14 clear weather /
   light requests, same type and same refusal behavior, 13/14 identical
   calls (the one diff adds an optional `on` field at conf 0.116 — a true
   tie).
3. **Residual divergences are quantization-sensitive near-ties**: 15 cases
   differ only in unconstrained reasoning wording; 3 flip the gate boundary
   where the two confidence estimators disagree (e.g. 0.158 vs 0.044); 1
   flips between EOS and `<tool_call>[]` (respond vs suppressed call —
   behaviorally identical: the agent stops either way).
4. No panics or one-sided errors on either side; the pure engine is
   byte-reproducible across repeat runs (guarded in the test).

## Design choices

**Two backends, one interface.** `needle.Engine` + `Options`/`Response` live in
the root package; backends are subpackages, so the import picks the policy.
`engine/cgo` pins you to a platform triple + libc++; `engine/pure` builds under
`CGO_ENABLED=0` anywhere. Keeping both was also a development strategy: the
cgo binding is ground truth, and the pure port doubles as a differential
oracle against the JAX reference.

**Weights as bytes, model as `go:embed`.** `Options.Weights []byte` matches the
engine ABI (`needle_load(cact, n)` reads memory in place) and lets
`model.Needle3` ship the whole 33.7 MB model inside the binary — no runtime
file resolution, one self-contained demo CLI.

**Python-mirrored surface.** `New/Complete/Run/Reset/Embed` and the JSON
`Response` keys (`function_calls`, `suppressed_calls`, …) mirror
`cactus-needle`, so ported examples read 1:1; the Go deltas are idiomatic:
`Tool.Fn` callbacks instead of dict dispatch, `error` returns instead of
exception-ish dicts, `ParseResponse` for wire JSON.

**Process-global semantics preserved.** `needle.h` declares the engine one
model per process, non-thread-safe. The cgo backend enforces that (mutex +
one live agent per kind); the pure backend keeps the same contract for API
symmetry even though it *could* host several agents (each would hold its own
~1.1 GB dequant copy — see improvements).

**FP32 eager dequant in the pure engine.** All CQ/FP16 records are decoded to
FP32 at `New`, making the per-step matvecs flat, memory-bound, and trivially
parallel (rows split across goroutines). Chose simplicity + fidelity over
memory: the result is closer to the JAX truth than the C engine (which runs
fp8-quantized KV/attention internally) at the cost of RSS. Everything runs
greedy argmax; the C engine's sampler knobs are not documented, so none were
invented.

**Grammar by parse-and-validate.** After `<tool_call>`, every candidate token
is checked by cloning the mini-JSON state and feeding the token's text
(~8192 clones × a few bytes ≈ 0.1 ms per step). Exactness over precomputed
char-class masks: a token is legal iff the whole span parses legal under the
schema (keys ∈ properties, names ∈ loaded tools, `</tool_call>` only after the
array closes). The reasoning phase stays free.

**Confidence definition.** Engine reports ≈ min(geometric mean of greedy token
probabilities, sigmoid(probe head)) — the geometric mean (not the raw product)
is what keeps long turns out of the 1e-30 basement and matches the engine's
band; see the 0.528 vs 0.529 check above.

## Where to improve next

- **Prefill batching**: the pure engine advances one token at a time even for
  prompts; a batched prefill (and per-layer pipeline parallelism) is the
  biggest latency lever (~3 s `Complete` today, most of it prompt).
- **Memory**: engram tables dominate the 1.1 GB RSS; keeping them FP16 (or
  dequant-on-access) roughly halves it. A shared dequantized `Model` handle
  would also allow multiple concurrent agents without duplicating weights.
- **Grammar completeness**: constrain the reasoning phase (the engine does),
  validate value *types* against the schema (`temperature: float`), and
  support multi-call arrays / streaming callbacks.
- **Numerics**: an optional fp8-KV emulation path would let the pure engine
  reproduce the C engine token-for-token instead of ~word-for-word.
- **Coverage**: the windows/arm64 targets are compile-checked only — CI should
  run the differential test there (the seeded corpus landed as
  `fuzzdiff_test.go`); darwin has no upstream static lib, so it is
  pure-engine-only forever unless Cactus ships one. Tokenizer round-trip
  fuzzing against the upstream `RefTokenizer` and expanding the corpus to
  random toolsets remain open.
- **Confidence estimator**: the engine's internal estimator is a black box;
  our geometric-mean + probe min tracks it (mean delta 0.041) but flips the
  suppression gate at the boundary on ~2 prompts in 35. Matching it exactly
  would require probing the engine with engineered logit patterns.
- **API sugar**: sampling options (temperature/top-p), populated
  `PrefillTPS`/`DecodeTPS`, and a `Kind`-based multi-model registry mirroring
  `needle_models()` when more checkpoints land.

## Repository layout

```
needle.go            shared API: Options, Tool, Response, Engine, Run loop
model/               needle3.cact + go:embed
engine/pure/         pure-Go runtime (cact.go, tokenizer.go, model.go,
                     forward.go, grammar.go, pure.go)
engine/cgo/          libneedle.a bindings, vendored needle.h + libs/<triple>/
cmd/needle-pure      demo CLI, pure engine
cmd/needle-cgo       demo CLI, prebuilt engine
```

## Upstream

- Engine + architecture: <https://github.com/cactus-compute/needle>
- Weights + prebuilt libs: <https://huggingface.co/Cactus-Compute/needle3>
- License: Apache-2.0 (upstream); vendored `needle.h` / `libneedle.a` /
  `needle3.cact` belong to Cactus Compute.
