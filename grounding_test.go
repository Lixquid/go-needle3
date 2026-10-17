package needle

// Tests for the strict grounding port in grounding.go. The cases mirror
// upstream tests/test_grounding.py, which pins the reference behaviour.

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestWithDateFact(t *testing.T) {
	now := time.Date(2026, 7, 21, 14, 30, 0, 0, time.Local)
	if got := WithDateFact("", now); got != "date: 2026-07-21 Tue 14:30" {
		t.Errorf("empty system = %q", got)
	}
	if got := WithDateFact("locale: en-US", now); got != "date: 2026-07-21 Tue 14:30; locale: en-US" {
		t.Errorf("facts = %q", got)
	}
	for _, already := range []string{"date: 2026-07-21 Tue 14:30", "locale: en-US; date: Tue"} {
		if got := WithDateFact(already, now); got != already {
			t.Errorf("%q must pass through, got %q", already, got)
		}
	}
	// A key that merely ends in "date" must not swallow the fact.
	for _, key := range []string{"user: Sam; last update: never", "candidate: Sam"} {
		if got := WithDateFact(key, now); !strings.HasPrefix(got, "date: ") {
			t.Errorf("%q should gain the date fact, got %q", key, got)
		}
	}
	if got := WithDateFact(`{"user":"Sam"}`, now); got != `{"user":"Sam"}` {
		t.Errorf("JSON facts should pass through, got %q", got)
	}
	if got := WithDateFact("meeting on 2026-08-02", now); got != "meeting on 2026-08-02" {
		t.Errorf("an ISO stamp already counts as a date, got %q", got)
	}
}

// extractSchema is the invoice shape used by the upstream grounding tests.
var extractSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"vendor":   map[string]any{"type": "string"},
		"total":    map[string]any{"type": "number"},
		"due_date": map[string]any{"type": "string", "format": "date"},
	},
}

func args(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestSourceYears(t *testing.T) {
	cases := []struct {
		text string
		want []int64
	}{
		{"due on 5th September 2031", []int64{2031}},
		{"due on 2024-05-06", []int64{2024}},
		{"due on 5/6/24", nil},
		{"due next Friday", nil},
		{"due on 5 June 19:30", nil},
		{"due on June 5 7 pm", nil},
		{"the March 2019 order", []int64{2019}},
		{"in the year 1999", []int64{1999}},
		{"date: 2026-07-21 Tue 14:30", []int64{2026}},
	}
	for _, c := range cases {
		got := sourceYears(c.text)
		if len(got) != len(c.want) {
			t.Errorf("sourceYears(%q) = %v, want %v", c.text, got, c.want)
			continue
		}
		for _, y := range c.want {
			if !got[y] {
				t.Errorf("sourceYears(%q) = %v, missing %d", c.text, got, y)
			}
		}
	}
}

func TestValidateExtractionDates(t *testing.T) {
	const text = "Send an invoice to Acme due on 5th September 2031"

	err := validateExtraction(text, extractSchema, args(t, `{"vendor":"Acme","due_date":"2026-09-05"}`), nil, "")
	var exErr *ExtractionError
	if !errors.As(err, &exErr) {
		t.Fatalf("a contradicting year must fail, got %v", err)
	}
	if !reflect.DeepEqual(exErr.Paths, []string{"due_date"}) {
		t.Errorf("paths = %v, want [due_date]", exErr.Paths)
	}
	if err := validateExtraction(text, extractSchema, args(t, `{"vendor":"Acme","due_date":"2031-09-05"}`), nil, ""); err != nil {
		t.Errorf("a grounded year must pass: %v", err)
	}
	// No year in the input means nothing to check against.
	if err := validateExtraction("Send Acme an invoice due next Friday", extractSchema,
		args(t, `{"vendor":"Acme","due_date":"2026-09-05"}`), nil, ""); err != nil {
		t.Errorf("without a source year nothing is checked: %v", err)
	}
	// A relative request is licensed by the date fact the engine saw.
	if err := validateExtraction("Invoice from Acme for the March 2019 order, due tomorrow",
		extractSchema, args(t, `{"vendor":"Acme","due_date":"2026-07-22"}`), nil,
		"date: 2026-07-21 Tue 14:30"); err != nil {
		t.Errorf("the date fact should license tomorrow: %v", err)
	}
	// Same text without the fact: 2026 is not licensed by "March 2019".
	if err := validateExtraction("Invoice from Acme for the March 2019 order, due tomorrow",
		extractSchema, args(t, `{"vendor":"Acme","due_date":"2026-07-22"}`), nil, ""); err == nil {
		t.Error("without a date fact the year must fail")
	}
}

func TestValidateExtractionNumbers(t *testing.T) {
	const text = "Invoice from Acme Corp, $1,200.00, due 2026-09-01"
	flagged := &Validation{Ungrounded: []string{"Invoice.total"}}

	// A thousands-separated number in the input clears the engine's flag.
	if err := validateExtraction(text, extractSchema, args(t, `{"vendor":"Acme","total":1200}`), flagged, ""); err != nil {
		t.Errorf("1,200.00 should ground 1200: %v", err)
	}
	if err := validateExtraction("Invoice from Acme Corp, $1200.00, due 2026-09-01", extractSchema,
		args(t, `{"vendor":"Acme","total":1200}`), flagged, ""); err != nil {
		t.Errorf("an unseparated number should ground 1200: %v", err)
	}
	// A fabricated number stays rejected.
	if err := validateExtraction(text, extractSchema, args(t, `{"vendor":"Acme","total":9999}`), flagged, ""); err == nil {
		t.Error("9999 is not in the input")
	}
	// A number absent from the input entirely.
	if err := validateExtraction("Invoice from Acme Corp, due 2026-09-01", extractSchema,
		args(t, `{"vendor":"Acme","total":1200}`), flagged, ""); err == nil {
		t.Error("1200 is not in the input")
	}
	// Digit runs inside a longer number do not count.
	for _, src := range []string{
		"Invoice from Acme Corp, $12,000.00, due 2026-09-01",
		"Invoice from Acme Corp, $11,200.00, due 2026-09-01",
		"Invoice from Acme Corp, $1,2000.00, due 2026-09-01",
	} {
		if err := validateExtraction(src, extractSchema, args(t, `{"total":1200}`), flagged, ""); err == nil {
			t.Errorf("%q must not ground 1200", src)
		}
	}
	// A date hyphen is not a minus sign, and a range dash is not either.
	for _, v := range []string{"-9", "-1", "-1400"} {
		if err := validateExtraction(text, extractSchema, args(t, `{"total":`+v+`}`), flagged, ""); err == nil {
			t.Errorf("%s must not be read out of a date separator", v)
		}
	}
	if err := validateExtraction("Invoice from Acme Corp, -1,200.00, due 2026-09-01", extractSchema,
		args(t, `{"total":-1200}`), flagged, ""); err != nil {
		t.Errorf("a written negative should ground -1200: %v", err)
	}
	if err := validateExtraction("Invoice from Acme Corp, $0.00, due 2026-09-01", extractSchema,
		args(t, `{"total":0}`), flagged, ""); err != nil {
		t.Errorf("0 should ground 0: %v", err)
	}
	// A nested leaf is grounded (and reported) by its own path.
	nested := map[string]any{"type": "object", "properties": map[string]any{
		"items": map[string]any{"type": "array", "items": map[string]any{
			"type":       "object",
			"properties": map[string]any{"price": map[string]any{"type": "number"}},
		}},
	}}
	if err := validateExtraction("Order with one item priced $1,100.00", nested,
		args(t, `{"items":[{"price":1100}]}`),
		&Validation{Ungrounded: []string{"Order.items[0].price"}}, ""); err != nil {
		t.Errorf("nested separated number: %v", err)
	}
	// A non-numeric flag cannot be cleared by the numbers in the input.
	err := validateExtraction(text, extractSchema, args(t, `{"vendor":"Acme","total":1200}`),
		&Validation{Ungrounded: []string{"Invoice.vendor"}}, "")
	if err == nil || !strings.Contains(err.Error(), "vendor") {
		t.Errorf("a non-numeric flag must stand, got %v", err)
	}
}

func TestValidateExtractionNegation(t *testing.T) {
	err := validateExtraction("no invoice needed", extractSchema, args(t, `{"vendor":"Acme"}`),
		&Validation{Negation: true}, "")
	var exErr *ExtractionError
	if !errors.As(err, &exErr) {
		t.Fatalf("a negation must fail, got %v", err)
	}
	if !strings.Contains(err.Error(), "negated request") {
		t.Errorf("message = %v", exErr)
	}
}

func TestValidateExtractionRefAndNestedDate(t *testing.T) {
	root := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"line": map[string]any{"$ref": "#/$defs/Line"},
		},
		"$defs": map[string]any{
			"Line": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"paid": map[string]any{
						"anyOf": []any{
							map[string]any{"type": "string", "format": "date"},
							map[string]any{"type": "null"},
						},
					},
				},
			},
		},
	}
	bad := args(t, `{"line":{"paid":"1999-01-02"}}`)
	if err := validateExtraction("due 5th September 2031", root, bad, nil, ""); err == nil {
		t.Error("a $ref/anyOf date must be checked")
	}
	ok := args(t, `{"line":{"paid":"2031-01-02"}}`)
	if err := validateExtraction("due 5th September 2031", root, ok, nil, ""); err != nil {
		t.Errorf("grounded nested date: %v", err)
	}
}

func TestParseResponseCarriesValidation(t *testing.T) {
	// The C engine attaches its grounding report to the turn; Extract folds it
	// into the strict check, so ParseResponse must not drop it.
	payload := `{"type":"call","success":true,"confidence":0.9,` +
		`"function_calls":[{"name":"Invoice","arguments":{"vendor":"Acme","due_date":"2026-09-05"}}],` +
		`"validation":{"ungrounded":["Invoice.vendor"],"negation":false}}`
	r, err := ParseResponse(payload)
	if err != nil {
		t.Fatal(err)
	}
	if r.Validation == nil {
		t.Fatalf("validation dropped: %+v", r)
	}
	if !reflect.DeepEqual(r.Validation.Ungrounded, []string{"Invoice.vendor"}) {
		t.Errorf("ungrounded = %v", r.Validation.Ungrounded)
	}
	ex, err := ExtractionFromResponse(r, Schema{Name: "Invoice", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"vendor": map[string]any{"type": "string"}},
	}}, "bill Acme for the March 2019 order", "", true)
	if err == nil {
		t.Error("the engine's own flag must survive into the strict check")
	} else if ex == nil || ex.Validation == nil {
		t.Errorf("extraction still carries the report: %+v", ex)
	}
	// An engine that reports nothing marshals nothing.
	if b, _ := json.Marshal(&Response{}); strings.Contains(string(b), "validation") {
		t.Errorf("empty response should omit validation: %s", b)
	}
}

// TestNumberTokens pins the hand-rolled scanner that stands in for the
// reference's lookaround-based regex: a literal may not start after a word
// character, "." or ",", may not run into a digit, and the thousands-grouped
// branch gives up (leaving the plain branch) when a group is incomplete.
func TestNumberTokens(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"$1,200.00", []string{"1200"}},
		{"$1200.00", []string{"1200"}},
		{"$1,2000.00", []string{"1"}}, // incomplete group: the plain branch mints 1
		{"$12,34", []string{"12"}},    // "34" is preceded by a comma, so it is not a literal
		{"$1,200.00-1,400.00", []string{"1200", "1400"}},
		{"2026-09-01", []string{"2026", "9", "1"}},
		{"-1,200.00", []string{"-1200"}},
		{"a1,200", []string{}}, // a word character precedes the digits
		{"x.5", []string{}},    // "." precedes them as well
		{"y+7", []string{"7"}},
		{"0.10 007 -0", []string{"1/10", "7", "0"}},
		{"no numbers", []string{}},
	}
	for _, c := range cases {
		var got []string
		for _, tok := range numberTokens(c.text) {
			if k := normalizeNumber(tok); k != "" {
				got = append(got, k)
			}
		}
		if len(got) == 0 && len(c.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("numberTokens(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestSourceNumbersSkipsDateFacts(t *testing.T) {
	// The injected date fact is context, not a quantity: its hyphens must not
	// mint negatives, and its digits must not ground a stray total.
	nums := sourceNumbers("Invoice for $1,200", "date: 2026-09-01 Tue 08:00")
	if nums["1200"] != true {
		t.Error("the amount in the text should be a source number")
	}
	if _, ok := nums["-2026"]; ok {
		t.Error("date facts must be stripped before scanning")
	}
	if len(sourceNumbers("")) != 0 {
		t.Error("empty input has no numbers")
	}
}

func TestValidateGroundingPublic(t *testing.T) {
	schema := Schema{Name: "Invoice", Parameters: extractSchema}

	// Ungrounded date, reported through the public entry point.
	err := ValidateGrounding("Send an invoice due on 5th September 2031", schema,
		args(t, `{"due_date":"1999-01-02"}`), "", nil)
	var exErr *ExtractionError
	if !errors.As(err, &exErr) {
		t.Fatalf("want an *ExtractionError, got %v", err)
	}
	if !strings.Contains(err.Error(), "due_date") {
		t.Errorf("message = %v", err)
	}
	// The tool-entry form unwraps "parameters" before walking.
	entry := map[string]any{"name": "Invoice", "parameters": extractSchema}
	if err := ValidateGrounding("due on 5th September 2031", entry,
		args(t, `{"due_date":"2031-09-05"}`), "", nil); err != nil {
		t.Errorf("tool-entry schema: %v", err)
	}
	// A schema with no parameter object cannot be walked.
	if err := ValidateGrounding("text", Schema{}, args(t, `{}`), "", nil); err == nil {
		t.Error("a schema without parameters must error")
	}
	// The engine's report folds in on this path too.
	if err := ValidateGrounding("no invoice needed", schema, args(t, `{"vendor":"Acme"}`),
		"", &Validation{Negation: true}); err == nil {
		t.Error("negation from the engine report must fail")
	}
}

func TestEngineUngroundedPaths(t *testing.T) {
	paths, negation := engineUngroundedPaths(&Validation{
		Ungrounded: []string{"Invoice.due_date", "Order.items[0].price", "bare"},
		Negation:   true,
	})
	if negation != true {
		t.Error("negation should pass through")
	}
	want := map[string]bool{"due_date": true, "items[0].price": true, "bare": true}
	if len(paths) != len(want) {
		t.Fatalf("paths = %v", paths)
	}
	for _, p := range paths {
		if !want[p] {
			t.Errorf("unexpected path %q", p)
		}
	}
	if paths, _ := engineUngroundedPaths(nil); paths != nil {
		t.Error("no report means no paths")
	}
}
