// Package cgo is a Go binding to the prebuilt Needle 3 C/C++ engine
// (libneedle.a from https://huggingface.co/Cactus-Compute/needle3).
//
// The engine keeps one process-global, non-thread-safe model per kind.
// This package serializes calls with a mutex and enforces a single live
// Agent per process; create a new Agent after Close.
//
// Build requirements: a C compiler plus the C++ runtime the prebuilt
// archives were linked against (libc++ / libc++abi). See the repository
// README for the per-platform matrix.
package cgo

/*
#include <stdlib.h>
#include <string.h>
#include "needle.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/Zetaphor/needle-go"
)

// Engine implements needle.Engine.
var _ needle.Engine = (*Agent)(nil)

var (
	globalMu sync.Mutex
	loaded   *Agent
)

// Agent is a Needle 3 agent bound to one toolset.
type Agent struct {
	model   unsafe.Pointer // C copy of the .cact bytes, kept alive for the engine
	buf     unsafe.Pointer // reusable output buffer
	bufSize C.int

	maxNewTokens int
	maxSteps     int
	tools        []needle.Tool
	closed       bool
}

// New loads the model and initializes the toolset. opts.Weights must hold the
// .cact model bytes (see the model package for the embedded copy).
func New(opts needle.Options) (*Agent, error) {
	globalMu.Lock()
	defer globalMu.Unlock()

	if loaded != nil {
		return nil, errors.New("needle/cgo: an Agent is already loaded (the engine holds one process-global model)")
	}
	if len(opts.Weights) == 0 {
		return nil, errors.New("needle/cgo: Options.Weights is empty (pass the .cact model bytes)")
	}
	opts = opts.WithDefaults()

	tools, err := opts.ToolsJSON()
	if err != nil {
		return nil, err
	}

	model := C.CBytes(opts.Weights)
	if model == nil {
		return nil, errors.New("needle/cgo: out of memory copying model weights")
	}
	if rc := C.needle_load((*C.uchar)(model), C.ulonglong(len(opts.Weights))); rc < 0 {
		C.free(model)
		return nil, fmt.Errorf("needle/cgo: needle_load failed (code %d): %s", int(rc), lastError())
	}

	// Zeroed buffer so the C string is always terminated even if the engine
	// writes exactly capacity bytes.
	buf := C.calloc(1, C.size_t(opts.BufferSize))
	if buf == nil {
		C.free(model)
		return nil, errors.New("needle/cgo: out of memory for output buffer")
	}

	cs := C.CString(opts.System)
	defer C.free(unsafe.Pointer(cs))
	ctools := C.CString(string(tools))
	defer C.free(unsafe.Pointer(ctools))
	if rc := C.needle_init(cs, ctools, nil); rc < 0 {
		C.free(buf)
		C.free(model)
		return nil, fmt.Errorf("needle/cgo: needle_init failed (code %d): %s (system prompt + tools may exceed the context window)", int(rc), lastError())
	}

	a := &Agent{
		model:        model,
		buf:          buf,
		bufSize:      C.int(opts.BufferSize),
		maxNewTokens: opts.MaxNewTokens,
		maxSteps:     opts.MaxSteps,
		tools:        opts.Tools,
	}
	loaded = a
	return a, nil
}

// Complete answers one turn. Feed tool results back as a JSON string to
// continue the conversation.
func (a *Agent) Complete(text string) (*needle.Response, error) {
	globalMu.Lock()
	defer globalMu.Unlock()
	if a.closed {
		return nil, errors.New("needle/cgo: agent is closed")
	}

	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	rc := C.needle_complete(ctext, nil, 0, C.int(a.maxNewTokens),
		(*C.char)(a.buf), a.bufSize)
	if rc < 0 {
		return nil, fmt.Errorf("needle/cgo: needle_complete failed (code %d): %s", int(rc), lastError())
	}
	return needle.ParseResponse(C.GoString((*C.char)(a.buf)))
}

// Run executes the full agentic loop using Tool.Fn handlers.
func (a *Agent) Run(query string) (*needle.Response, error) {
	// runLoop calls Complete; take no lock around the loop itself, but guard
	// the agent against Close mid-loop.
	globalMu.Lock()
	if a.closed {
		globalMu.Unlock()
		return nil, errors.New("needle/cgo: agent is closed")
	}
	globalMu.Unlock()

	r, err := needle.RunLoopFunc(a.Complete, a.tools, query, a.maxSteps)
	return r, err
}

// Reset rewinds the conversation, keeping the tools loaded.
func (a *Agent) Reset() error {
	globalMu.Lock()
	defer globalMu.Unlock()
	if a.closed {
		return errors.New("needle/cgo: agent is closed")
	}
	C.needle_reset()
	return nil
}

// Embed returns the model's embedding for a text.
func (a *Agent) Embed(text string) ([]float32, error) {
	globalMu.Lock()
	defer globalMu.Unlock()
	if a.closed {
		return nil, errors.New("needle/cgo: agent is closed")
	}

	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	dim := C.needle_embed(ctext, nil, 0, nil, 0)
	if dim <= 0 {
		return nil, fmt.Errorf("needle/cgo: needle_embed failed (code %d): %s", int(dim), lastError())
	}
	out := make([]float32, int(dim))
	rc := C.needle_embed(ctext, nil, 0, (*C.float)(unsafe.Pointer(&out[0])), dim)
	if rc < 0 {
		return nil, fmt.Errorf("needle/cgo: needle_embed failed (code %d): %s", int(rc), lastError())
	}
	return out, nil
}

// Models reports which model kinds are loaded in this process: a bitwise
// combination of needle.ModelText / needle.ModelSpeech semantics from the C
// header (1 = text, 2 = speech).
func Models() int {
	globalMu.Lock()
	defer globalMu.Unlock()
	return int(C.needle_models())
}

// Close releases this agent. The engine itself keeps its state until the
// process exits; Close frees the Go-owned copies and allows a new New call.
func (a *Agent) Close() error {
	globalMu.Lock()
	defer globalMu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	C.free(a.model)
	C.free(a.buf)
	if loaded == a {
		loaded = nil
	}
	return nil
}

func lastError() string {
	p := C.needle_last_error()
	if p == nil {
		return "unknown error"
	}
	return C.GoString(p)
}
