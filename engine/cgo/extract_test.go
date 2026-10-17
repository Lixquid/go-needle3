package cgo

import (
	"errors"
	"testing"

	needle "github.com/lixquid/go-needle3"
	nmodel "github.com/lixquid/go-needle3/model"
)

// contact is the extraction record used by the upstream test suite.
type contact struct {
	Name  string `json:"name" description:"the person's full name"`
	Email string `json:"email" description:"their email address"`
}

func extractOptions() needle.Options {
	return needle.Options{Weights: nmodel.Needle3, System: "date: 2026-10-08 Thu 09:41; device: laptop"}
}

// TestExtractContact is the end-to-end path on the prebuilt engine: schema in,
// typed record out, with the strict grounding check folded in. Assertions stay
// soft about what the model chooses to emit — a missing field is a model
// outcome, not a port bug — and hard about the plumbing.
//
// These tests stay sequential: the engine holds one process-global model, so
// a bound agent must be closed before the next Extract builds one.
func TestExtractContact(t *testing.T) {
	ex, err := Extract("John Doe can be reached at john@doe.com", needle.SchemaOf[contact](), extractOptions())
	if errors.Is(err, needle.ErrNoExtraction) {
		t.Skip("the engine returned no record for this text")
	}
	var exErr *needle.ExtractionError
	if err != nil && !errors.As(err, &exErr) {
		t.Fatalf("extract: %v", err)
	} else if err != nil {
		t.Logf("strict rejection: %v", exErr)
	}
	if ex == nil || len(ex.Arguments) == 0 {
		t.Fatalf("a call came back with no arguments: %+v", ex)
	}

	var c contact
	if err := ex.Decode(&c); err != nil {
		t.Fatalf("decode: %v", err)
	}
	t.Logf("name=%q email=%q conf=%.3f suppressed=%v reasoning=%q",
		c.Name, c.Email, ex.Confidence, ex.Suppressed, ex.Reasoning)
	if c.Email == "" {
		t.Logf("email omitted by the model: %v", ex.Arguments)
	}
}

// TestExtractAsTyped checks the generic entry point, which names the schema
// after the type and decodes in one step.
func TestExtractAsTyped(t *testing.T) {
	c, err := ExtractAs[contact]("Please email Jane Smith at jane@smith.example", extractOptions())
	switch {
	case errors.Is(err, needle.ErrNoExtraction):
		t.Skip("the engine returned no record for this text")
	case err != nil:
		var exErr *needle.ExtractionError
		if !errors.As(err, &exErr) || c == nil {
			t.Fatalf("ExtractAs: %v", err)
		}
	}
	if c == nil {
		t.Fatal("a nil record with no error")
	}
	t.Logf("contact = %+v", *c)
}

// TestExtractTurnOnBoundAgent covers the reuse path: bind the schema once, then
// extract turn after turn, resetting between unrelated texts.
func TestExtractTurnOnBoundAgent(t *testing.T) {
	a, err := New(extractOptions().ForExtraction(needle.SchemaOf[contact]()))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	for _, text := range []string{"John Doe: john@doe.com", "write to bob@corp.io, that is Bob Jones"} {
		ex, err := a.ExtractTurn(text)
		if err != nil {
			t.Logf("%q: %v", text, err)
		} else {
			t.Logf("%q: %v conf=%.3f", text, ex.Arguments, ex.Confidence)
		}
		if err := a.Reset(); err != nil {
			t.Fatal(err)
		}
	}
}
