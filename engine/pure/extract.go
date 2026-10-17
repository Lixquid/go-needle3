package pure

// One-shot structured extraction on the pure-Go engine.
//
// Extraction is tool calling with a single record-shaped tool: the schema is
// bound as the toolset, the text is the one turn, and the call's arguments are
// the answer. These helpers create that agent, run the turn and close it,
// mirroring the Python package's needle.extract(); the shared half of the
// behaviour (schema shapes, grounding, decoding) lives in the root package.

import (
	"crypto/sha256"
	"errors"
	"sync"

	"github.com/lixquid/go-needle3"
)

// Extract extracts schema from text in one turn and returns the record as
// generic arguments. schema accepts a Go struct type value, a JSON-schema map
// (or JSON text), a needle.Schema or a needle.Tool — see needle.SchemaFrom.
//
// A fresh agent is built per call, but the dequantized model is cached per
// weight set (see ReleaseModelCache), so repeated extraction pays the
// dequantization once and costs one turn — a few seconds on this engine.
//
// On a call the model withheld below its confidence gate the values are still
// returned, with Extraction.Suppressed set; check Extraction.Confidence. When
// the model returns no call at all the error is needle.ErrNoExtraction.
//
//	opts := needle.Options{Weights: model.Needle3}
//	ex, err := pure.Extract("John Doe can be reached at john@doe.com",
//		map[string]any{"name": "Contact", "parameters": map[string]any{
//			"type": "object",
//			"properties": map[string]any{
//				"name":  map[string]any{"type": "string"},
//				"email": map[string]any{"type": "string"},
//			},
//			"required": []string{"name", "email"},
//		}}, opts)
func Extract(text string, schema any, opts needle.Options) (*needle.Extraction, error) {
	return needle.ExtractWith(newCached, text, schema, opts)
}

// ExtractAs is the typed form of Extract: the schema is derived from T with
// needle.SchemaOf and the arguments are decoded into a new T.
//
//	type Contact struct {
//		Name  string `json:"name" description:"full name"`
//		Email string `json:"email" description:"email address"`
//	}
//	contact, err := pure.ExtractAs[Contact]("John Doe: john@doe.com", opts)
//
// The schema name is T's name, which the model sees as the record name; use
// Extract with an explicit needle.Schema when that matters. Fields the model
// leaves out stay zero: the engine emits what it can ground, and a missing
// required field is reported by a low Confidence rather than an error.
func ExtractAs[T any](text string, opts needle.Options) (*T, error) {
	return needle.ExtractAsWith[T](newCached, text, opts)
}

// newCached builds an Agent on the cached dequantized model for these weights,
// so a loop of Extract calls does not dequantize once per call. The model is
// read-only after loading and safe to share across agents.
func newCached(opts needle.Options) (needle.Engine, error) {
	m, err := cachedModel(opts.Weights)
	if err != nil {
		return nil, err
	}
	return newAgent(m, opts.WithDefaults())
}

var (
	modelCacheMu sync.Mutex
	modelCache   = map[[sha256.Size]byte]*model{}
)

// cachedModel returns the dequantized model for weights, loading it once per
// distinct weight set. The key is a hash of the bytes, so the same embedded
// needle3.cact always hits the same entry (~40 ms to hash, against ~1.5 s to
// dequantize).
func cachedModel(weights []byte) (*model, error) {
	if len(weights) == 0 {
		return nil, errors.New("pure: Options.Weights is empty (pass the .cact model bytes)")
	}
	key := sha256.Sum256(weights)

	modelCacheMu.Lock()
	defer modelCacheMu.Unlock()
	if m, ok := modelCache[key]; ok {
		return m, nil
	}
	m, err := loadModel(weights)
	if err != nil {
		return nil, err
	}
	modelCache[key] = m
	return m, nil
}

// ReleaseModelCache frees the models Extract has dequantized. Callers that
// extract once per process may prefer it to keeping ~1.1 GB of FP32 weights
// resident; a later Extract just loads the model again.
func ReleaseModelCache() {
	modelCacheMu.Lock()
	defer modelCacheMu.Unlock()
	modelCache = map[[sha256.Size]byte]*model{}
}

// ExtractTurn is the agent-reusing path: bind the schema once with
// Options.ForExtraction at New time, then extract turn after turn on the one
// loaded model and decode each result. Call Reset between unrelated texts.
func (a *Agent) ExtractTurn(text string) (*needle.Extraction, error) {
	resp, err := a.Complete(text)
	if err != nil {
		return nil, err
	}
	sc := needle.Schema{}
	if len(a.opts.Tools) > 0 {
		t := a.opts.Tools[0]
		sc = needle.Schema{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
	}
	return needle.ExtractionFromResponse(resp, sc, text, a.opts.System, a.opts.StrictExtraction())
}
