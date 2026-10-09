// Package needle defines the shared API surface for Needle 3 inference in Go.
//
// Needle 3 (Cactus Compute) is a tiny on-device model for tool calling:
// text goes in, a JSON tool call comes back. Two interchangeable backends
// implement the Engine interface defined here:
//
//   - engine/cgo:   bindings to the prebuilt C/C++ engine (libneedle.a)
//   - engine/pure:  a pure-Go port of the runtime (no cgo, no libc)
//
// The mirror of the Python cactus-needle package:
//
//	python                         go
//	needle.Needle(tools=[...])     cgo.New(needle.Options{Tools: ...})
//	agent.complete(text)           agent.Complete(text)
//	agent.run(query)               agent.Run(query)
//	agent.reset()                  agent.Reset()
//	agent.embed(text)              agent.Embed(text)
//
// The underlying engine holds one process-global, non-thread-safe model per
// kind; backends serialize calls with a mutex and expect a single live Agent
// per process.
package needle

import (
	"encoding/json"
	"fmt"
)

// Tool describes one callable function exposed to the model, mirroring the
// raw JSON-schema form that the engine consumes. Fn is optional and only used
// by Run to execute calls automatically.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON-schema object, e.g.
	// {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}.
	// A nil value renders as an empty parameter object.
	Parameters map[string]any
	// Fn executes this tool during Run. Its result is fed back to the model
	// as JSON.
	Fn func(args map[string]any) (any, error)
}

// Call is a single function call chosen by the model.
type Call struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// Result records one executed call during Run.
type Result struct {
	Call   Call
	Output any
	Err    error
}

// Response is the outcome of one turn, mirroring the JSON object returned by
// the Python package's complete()/run().
type Response struct {
	// Type is "call" when the model wants tool calls (an empty FunctionCalls
	// list is the off-topic refusal), or "respond" when the loop is finished.
	Type          string `json:"type"`
	Success       bool   `json:"success"`
	Error         string `json:"error,omitempty"`
	ErrorCode     string `json:"error_code,omitempty"`
	FunctionCalls []Call `json:"function_calls"`
	// SuppressedCalls holds a call the engine withheld (low confidence or a
	// grounding gate); FunctionCalls is then empty.
	SuppressedCalls []Call `json:"suppressed_calls,omitempty"`
	Reasoning       string `json:"reasoning,omitempty"`
	// Confidence is a calibrated score in [0,1]. HasConfidence is false when
	// the loaded weights carry no calibration head.
	Confidence    float64 `json:"confidence"`
	HasConfidence bool    `json:"-"`
	PrefillTPS    float64 `json:"prefill_tps,omitempty"`
	DecodeTPS     float64 `json:"decode_tps,omitempty"`

	// Results holds the outputs of executed calls; populated by Run only.
	Results []Result `json:"-"`

	// Raw carries the raw generated text for this turn (pure backend) or the
	// engine's raw field, useful for debugging.
	Raw string `json:"raw,omitempty"`
}

// Options configures an Agent.
type Options struct {
	// Tools is the toolset bound to this agent. One toolset per session:
	// changing tools requires a new Agent.
	Tools []Tool
	// System is an optional environment-facts string, e.g.
	// "date: 2026-07-21 Tue 14:30; locale: en-US; device: phone".
	System string
	// Weights holds the .cact model bytes; required. Use github.com/.../model
	// for the embedded needle3.cact.
	Weights []byte
	// BufferSize sizes the engine's JSON output buffer. Default 65536.
	BufferSize int
	// MaxNewTokens caps each generated turn. Default 512.
	MaxNewTokens int
	// MaxSteps caps the Run agentic loop. Default 8.
	MaxSteps int
}

// Engine is implemented by both backends.
type Engine interface {
	// Complete answers one turn. Feed tool results back as a JSON string.
	Complete(text string) (*Response, error)
	// Run executes the full agentic loop: the model picks calls, matching
	// Tool.Fn handlers are executed, results are fed back, and the final
	// response carries the executed outputs in Response.Results.
	Run(query string) (*Response, error)
	// Reset rewinds the conversation, keeping the tools loaded.
	Reset() error
	// Embed returns the model's embedding for a text (3072 floats on the
	// shipped needle3.cact).
	Embed(text string) ([]float32, error)
	// Close releases the agent. A new Agent may be created afterwards.
	Close() error
}

// Defaults shared by the backends.
const (
	DefaultBufferSize   = 65536
	DefaultMaxNewTokens = 512
	DefaultMaxSteps     = 8
)

func (o Options) WithDefaults() Options {
	if o.BufferSize <= 0 {
		o.BufferSize = DefaultBufferSize
	}
	if o.MaxNewTokens <= 0 {
		o.MaxNewTokens = DefaultMaxNewTokens
	}
	if o.MaxSteps <= 0 {
		o.MaxSteps = DefaultMaxSteps
	}
	return o
}

// ParseResponse decodes the engine's per-turn JSON object into a Response,
// detecting whether a calibrated confidence was present.
func ParseResponse(data string) (*Response, error) {
	var r Response
	if err := json.Unmarshal([]byte(data), &r); err != nil {
		return nil, fmt.Errorf("needle: parse response: %w (payload: %.200s)", err, data)
	}
	var probe struct {
		Confidence *float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(data), &probe); err == nil && probe.Confidence != nil {
		r.Confidence = *probe.Confidence
		r.HasConfidence = true
	}
	return &r, nil
}

// ToolsJSON renders the tool list into the minified JSON array the engine
// consumes: [{"name":...,"description":...,"parameters":{...}}].
func (o Options) ToolsJSON() ([]byte, error) {
	if len(o.Tools) == 0 {
		return []byte("[]"), nil
	}
	out := make([]map[string]any, 0, len(o.Tools))
	for _, t := range o.Tools {
		entry := map[string]any{"name": t.Name, "description": t.Description}
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object"}
		}
		entry["parameters"] = params
		out = append(out, entry)
	}
	return json.Marshal(out)
}

// RunLoopFunc is the shared implementation of Engine.Run, driving a backend's
// Complete method and executing matching Tool.Fn handlers.
func RunLoopFunc(complete func(string) (*Response, error), tools []Tool, query string, maxSteps int) (*Response, error) {
	if maxSteps <= 0 {
		maxSteps = DefaultMaxSteps
	}
	byName := make(map[string]func(map[string]any) (any, error), len(tools))
	for _, t := range tools {
		if t.Fn != nil {
			byName[t.Name] = t.Fn
		}
	}
	r, err := complete(query)
	if err != nil {
		return nil, err
	}
	var results []Result
	for step := 1; step < maxSteps; step++ {
		if len(r.FunctionCalls) == 0 {
			break
		}
		next := make([]any, 0, len(r.FunctionCalls))
		for _, call := range r.FunctionCalls {
			fn, ok := byName[call.Name]
			if !ok {
				break // no handler: stop the loop, caller executes
			}
			out, ferr := fn(call.Arguments)
			results = append(results, Result{Call: call, Output: out, Err: ferr})
			next = append(next, out)
		}
		if len(next) != len(r.FunctionCalls) {
			break
		}
		payload, merr := json.Marshal(next)
		if merr != nil {
			return nil, fmt.Errorf("needle: marshal tool results: %w", merr)
		}
		r, err = complete(string(payload))
		if err != nil {
			return nil, err
		}
	}
	r.Results = results
	return r, nil
}
