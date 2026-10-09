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
  free, so the generated reasoning can occasionally drift where the engine's
  stricter internal grammar would not (`{"city":"turn on"}` vs
  `{"city":"kitchen"}`).
- Everything runs in FP32; the C engine quantizes KV/attention internally, so
  occasional token-level divergence is expected under identical prompts.
- The engine's process-global single-model constraint is preserved: one live
  agent per kind per process (backends guard with a mutex; the pure engine is
  internally parallel over matvecs).

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
