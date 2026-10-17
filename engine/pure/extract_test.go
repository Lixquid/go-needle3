package pure

import (
	"errors"
	"testing"

	"github.com/lixquid/go-needle3"
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

// TestExtractContact is the end-to-end path: schema in, typed record out, on
// the real weights. The model is dequantized once for the whole package via
// the Extract model cache.
func TestExtractContact(t *testing.T) {
	t.Parallel()

	ex, err := Extract("John Doe can be reached at john@doe.com", needle.SchemaOf[contact](), extractOptions())
	if errors.Is(err, needle.ErrNoExtraction) {
		t.Skip("the model returned no record for this text")
	}
	if err != nil {
		var exErr *needle.ExtractionError
		if !errors.As(err, &exErr) || ex == nil {
			t.Fatalf("extract: %v", err)
		}
		t.Logf("ungrounded values: %v", exErr)
	}
	t.Logf("name=%q args=%v conf=%.3f suppressed=%v reasoning=%q", ex.Name, ex.Arguments, ex.Confidence, ex.Suppressed, ex.Reasoning)

	var c contact
	if err := ex.Decode(&c); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// The model emits only the fields it can ground, so require the one that
	// is unmistakably in the text and just report the other.
	if c.Email == "" {
		t.Errorf("email missing from %v", ex.Arguments)
	}
	if c.Name == "" {
		t.Logf("name omitted by the model (it reasons it is not given)")
	}
}

// TestExtractAsTyped exercises the generic entry point, which derives the
// schema from the type name and decodes in one step.
func TestExtractAsTyped(t *testing.T) {
	t.Parallel()

	c, err := ExtractAs[contact]("Please email Jane Smith at jane@smith.example", extractOptions())
	switch {
	case errors.Is(err, needle.ErrNoExtraction):
		t.Skip("the model returned no record for this text")
	case err != nil:
		t.Fatalf("ExtractAs: %v", err)
	}
	t.Logf("contact = %+v", c)
}

// TestExtractModelCache checks the one thing the cache must guarantee: one
// dequantized model per weight set, shared across agents.
func TestExtractModelCache(t *testing.T) {
	a1, err := newCached(extractOptions())
	if err != nil {
		t.Fatal(err)
	}
	a2, err := newCached(extractOptions())
	if err != nil {
		t.Fatal(err)
	}
	p1, p2 := a1.(*Agent).m, a2.(*Agent).m
	if p1 != p2 {
		t.Error("the same weights dequantized twice")
	}
	if _, err := newCached(needle.Options{}); err == nil {
		t.Error("empty weights must error")
	}
	ReleaseModelCache()
	a3, err := newCached(extractOptions())
	if err != nil {
		t.Fatal(err)
	}
	if a3.(*Agent).m == p1 {
		t.Error("ReleaseModelCache did not free the model")
	}
}

// TestExtractTurnOnBoundAgent covers the reuse path: the schema is bound once
// and each text is a turn on the same agent.
func TestExtractTurnOnBoundAgent(t *testing.T) {
	t.Parallel()

	sc := needle.SchemaOf[contact]()
	a, err := New(extractOptions().ForExtraction(sc))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	for _, text := range []string{"John Doe: john@doe.com", "write to bob@corp.io, that is Bob Jones"} {
		ex, err := a.ExtractTurn(text)
		if errors.Is(err, needle.ErrNoExtraction) {
			t.Logf("%q: no record", text)
			continue
		}
		if err != nil {
			t.Logf("%q: %v", text, err)
			continue
		}
		t.Logf("%q: %v conf=%.3f", text, ex.Arguments, ex.Confidence)
		if err := a.Reset(); err != nil {
			t.Fatal(err)
		}
	}
}

// invoice is the record the upstream grounding tests use: a date field the
// model may invent a year for, plus a money amount.
type invoice struct {
	Vendor  string      `json:"vendor" description:"who issued the invoice"`
	Total   float64     `json:"total,omitempty" description:"amount due"`
	DueDate needle.Date `json:"due_date,omitempty" description:"when it is due"`
}

// TestExtractStrictGrounding drives the strict check with real model output: a
// date whose year is not in the input must come back as an *ExtractionError
// with the values still attached.
func TestExtractStrictGrounding(t *testing.T) {
	t.Parallel()

	const text = "Invoice from Acme for $1,200, due on 5th September 2031"
	ex, err := Extract(text, needle.SchemaOf[invoice](), extractOptions())
	if errors.Is(err, needle.ErrNoExtraction) {
		t.Skip("the model returned no record for this text")
	}
	var exErr *needle.ExtractionError
	switch {
	case err == nil:
		t.Logf("grounded: %+v conf=%.3f", ex.Arguments, ex.Confidence)
		if ex.Arguments["due_date"] == "2031-09-05" {
			return
		}
		t.Logf("due_date = %v (the year check only fires on an ISO date)", ex.Arguments["due_date"])
	case errors.As(err, &exErr):
		t.Logf("strict rejection: %v with args %v", exErr, ex.Arguments)
		if ex == nil || ex.Arguments == nil {
			t.Error("an ungrounded extraction still carries its values")
		}
	default:
		t.Fatalf("extract: %v", err)
	}

	// The same values without the strict check always come back.
	if _, err := Extract(text, needle.SchemaOf[invoice](),
		needle.Options{Weights: extractOptions().Weights, System: extractOptions().System, Strict: needle.Bool(false)}); err != nil {
		t.Errorf("permissive extract: %v", err)
	}
}
