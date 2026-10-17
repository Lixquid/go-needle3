package cgo

// One-shot structured extraction on the prebuilt C/C++ engine — the same path
// the Python wheel takes, and the fast one here: the .cact is read in place, so
// a fresh agent costs ~0.1 s rather than a dequantization pass.

import (
	"github.com/lixquid/go-needle3"
)

// newEngine adapts New to needle.EngineFactory.
func newEngine(o needle.Options) (needle.Engine, error) { return New(o) }

// Extract extracts schema from text in one turn and returns the record as
// generic arguments. schema accepts a Go struct type value, a JSON-schema map
// (or JSON text), a needle.Schema or a needle.Tool — see needle.SchemaFrom.
//
// The engine holds one process-global model, so Extract needs an Agent slot of
// its own and fails while another Agent is live; use Agent.ExtractTurn to loop
// over many texts against one loaded model.
//
// A call the engine withheld below its confidence gate still yields its
// values, with Extraction.Suppressed set. The engine's own grounding report is
// carried in Extraction.Validation and folded into the strict check. When the
// model returns no call at all the error is needle.ErrNoExtraction.
func Extract(text string, schema any, opts needle.Options) (*needle.Extraction, error) {
	return needle.ExtractWith(newEngine, text, schema, opts)
}

// ExtractAs is the typed form of Extract: the schema is derived from T with
// needle.SchemaOf and the arguments are decoded into a new T.
//
//	type Contact struct {
//		Name  string `json:"name" description:"full name"`
//		Email string `json:"email" description:"email address"`
//	}
//	contact, err := cgo.ExtractAs[Contact]("John Doe: john@doe.com", opts)
func ExtractAs[T any](text string, opts needle.Options) (*T, error) {
	return needle.ExtractAsWith[T](newEngine, text, opts)
}

// ExtractTurn is the agent-reusing path: bind the schema as the agent's toolset
// at New time, then extract turn after turn on one loaded model. Call Reset
// between unrelated texts.
func (a *Agent) ExtractTurn(text string) (*needle.Extraction, error) {
	resp, err := a.Complete(text)
	if err != nil {
		return nil, err
	}
	var sc needle.Schema
	if len(a.tools) > 0 {
		t := a.tools[0]
		sc = needle.Schema{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
	}
	return needle.ExtractionFromResponse(resp, sc, text, a.system, a.strict)
}
