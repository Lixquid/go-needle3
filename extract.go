package needle

// Structured extraction.
//
// Extraction is tool calling with exactly one record-shaped tool: the schema
// goes into the prompt as the toolset, the text is the one turn, and the
// single call's arguments are the extracted record. The decode grammar
// guarantees the output parses, which is what makes extraction (and
// classification through enums) reliable.
//
// The backends expose the entry points (pure.Extract, cgo.Extract); this file
// holds the engine-independent half so both behave identically:
//
//	python                                   go
//	extract(text, Contact)                   pure.Extract(text, needle.SchemaOf[Contact]{}, opts)
//	                                         pure.ExtractAs[Contact](text, opts)
//	ExtractionValidationError                *needle.ExtractionError
//	needle._with_date_fact                   needle.WithDateFact
//
// A fresh agent is built per call because an engine binds one toolset for the
// life of an agent; callers extracting in a loop bind the schema once with
// Options.ForExtraction and drive Agent.ExtractTurn (or Engine.Complete plus
// ExtractionFromResponse), resetting between unrelated texts.

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"time"
)

// DefaultSchemaName is the tool name used for a schema that does not carry
// one. It shows up in the prompt and in the model's "name" field.
const DefaultSchemaName = "extract"

// Schema is the shape to extract to: one record, expressed as the JSON-schema
// object the engine consumes for a single tool.
type Schema struct {
	// Name is the record name offered to the model, e.g. "Invoice". Empty
	// becomes DefaultSchemaName.
	Name string
	// Description is an optional one-liner shown to the model; it is what
	// steers ambiguous fields.
	Description string
	// Parameters is the JSON-schema object for the record: typically
	// {"type":"object","properties":{...},"required":[...]}. A nil value
	// means an empty object.
	Parameters map[string]any
}

// SchemaProvider lets a type decide what JSON schema is generated for it.
// A pointer receiver works too: the method is called on a zero value.
type SchemaProvider interface {
	JSONSchema() map[string]any
}

// Described lets a record type supply the schema description, the Go stand-in
// for the docstring that pydantic extraction uses. It is called on a zero
// value, so it must not read the receiver's fields.
type Described interface {
	Description() string
}

// Date is a calendar date for extraction schemas: it accepts "2026-09-01"
// (the format the model emits for "format":"date") and RFC 3339 stamps, and
// renders as the plain date. encoding/json's time.Time rejects the bare date.
type Date time.Time

// Time returns the wrapped time value.
func (d Date) Time() time.Time { return time.Time(d) }

// String renders the date as "2006-01-02".
func (d Date) String() string { return time.Time(d).Format("2006-01-02") }

// MarshalJSON writes the "2006-01-02" form.
func (d Date) MarshalJSON() ([]byte, error) { return json.Marshal(d.String()) }

// UnmarshalJSON accepts "2006-01-02", an RFC 3339 stamp (the time part is
// dropped), or an empty string (the zero Date).
func (d *Date) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	if s == "" {
		*d = Date{}
		return nil
	}
	for _, layout := range []string{"2006-01-02", time.RFC3339, "2006-01-02T15:04:05", "2006-01-02 15:04:05"} {
		if t, err := time.Parse(layout, s); err == nil {
			*d = Date(t)
			return nil
		}
	}
	return fmt.Errorf("needle: cannot parse %q as a date", s)
}

// JSONSchema marks the field as a date, which is also what switches on the
// temporal grounding check in a strict Extract.
func (Date) JSONSchema() map[string]any {
	return map[string]any{"type": "string", "format": "date"}
}

// SchemaOf derives a Schema from a Go type using encoding/json conventions.
// See schema_reflect.go for the mapping and the struct tags it honours.
func SchemaOf[T any]() Schema {
	return schemaForType(reflect.TypeOf((*T)(nil)).Elem())
}

// SchemaFrom accepts the schema forms Extract takes:
//
//   - Schema (or *Schema)                      — used as given
//   - Tool                                     — name, description, parameters
//   - map[string]any                           — a tool entry with "parameters",
//     or a bare JSON-schema object
//   - string / []byte                          — the same two forms as JSON text
//   - reflect.Type, struct, *struct, map type  — derived like SchemaOf
func SchemaFrom(v any) (Schema, error) {
	switch s := v.(type) {
	case nil:
		return Schema{}, errors.New("needle: extract schema is nil")
	case Schema:
		return s.normalized(), nil
	case *Schema:
		if s == nil {
			return Schema{}, errors.New("needle: extract schema is nil")
		}
		return s.normalized(), nil
	case Tool:
		return Schema{Name: s.Name, Description: s.Description, Parameters: s.Parameters}.normalized(), nil
	case reflect.Type:
		if s == nil {
			return Schema{}, errors.New("needle: extract schema type is nil")
		}
		return schemaFromType(s)
	case map[string]any:
		return schemaFromMap(s).normalized(), nil
	case []byte:
		return schemaFromJSON(string(s))
	case string:
		return schemaFromJSON(s)
	}
	t := reflect.TypeOf(v)
	if t == nil {
		return Schema{}, errors.New("needle: extract schema is nil")
	}
	st := t
	for st.Kind() == reflect.Pointer {
		st = st.Elem()
	}
	switch st.Kind() {
	case reflect.Struct, reflect.Map:
		return schemaFromType(t)
	}
	return Schema{}, fmt.Errorf("needle: cannot use %T as an extract schema (want Schema, Tool, a JSON-schema map, JSON text, or a struct/map type)", v)
}

// Tool renders the schema as the single tool entry an Agent binds.
func (s Schema) Tool() Tool {
	s = s.normalized()
	return Tool{Name: s.Name, Description: s.Description, Parameters: s.Parameters}
}

// normalized fills in the defaults (record name, object parameter schema).
func (s Schema) normalized() Schema {
	if strings.TrimSpace(s.Name) == "" {
		s.Name = DefaultSchemaName
	}
	if s.Parameters == nil {
		s.Parameters = map[string]any{"type": "object"}
	}
	if _, ok := s.Parameters["type"]; !ok {
		s.Parameters = withType(s.Parameters)
	}
	return s
}

// withType returns a copy of params with "type":"object" set.
func withType(params map[string]any) map[string]any {
	out := make(map[string]any, len(params)+1)
	for k, v := range params {
		out[k] = v
	}
	out["type"] = "object"
	return out
}

func schemaFromMap(m map[string]any) Schema {
	if p, ok := m["parameters"].(map[string]any); ok {
		sc := Schema{Parameters: p}
		if n, ok := m["name"].(string); ok {
			sc.Name = n
		}
		if d, ok := m["description"].(string); ok {
			sc.Description = d
		}
		return sc
	}
	return Schema{Parameters: m}
}

func schemaFromJSON(text string) (Schema, error) {
	var m map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &m); err != nil {
		return Schema{}, fmt.Errorf("needle: parse extract schema JSON: %w", err)
	}
	return schemaFromMap(m).normalized(), nil
}

// Extraction is the outcome of one extraction turn.
type Extraction struct {
	// Schema is the shape that was extracted, defaults filled in.
	Schema Schema
	// Name is the record name the model used (normally Schema.Name).
	Name string
	// Arguments holds the extracted fields, keyed by schema property.
	Arguments map[string]any

	// Reasoning is the model's <think> text for the turn.
	Reasoning string
	// Confidence is a calibrated score in [0,1]; HasConfidence is false when
	// the loaded weights carry no calibration head.
	Confidence    float64
	HasConfidence bool
	// Suppressed is true when the engine withheld the call below its
	// confidence gate. Extraction still returns the withheld values; a low
	// Confidence is the signal to distrust them.
	Suppressed bool
	// Validation is the engine's own grounding report, when it reports one.
	// The cgo backend passes the engine's "validation" object through; the
	// pure backend has no engine-side flags and leaves it nil.
	Validation *Validation
	// Raw is the raw generated text for the turn, useful for debugging.
	Raw string

	// Response is the full turn the extraction came from.
	Response *Response
}

// ErrNoExtraction is returned when the model produced no call for the text:
// it read the input as off-topic, or answered instead of extracting.
var ErrNoExtraction = errors.New("needle: no extraction (the model returned no call)")

// Decode copies the extracted arguments into out, which must be a non-nil
// pointer, typically *T for the same type SchemaOf was given.
func (e *Extraction) Decode(out any) error {
	if e == nil || len(e.Arguments) == 0 {
		return ErrNoExtraction
	}
	if out == nil {
		return errors.New("needle: extract decode target is nil")
	}
	rv := reflect.ValueOf(out)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("needle: extract decode target must be a non-nil pointer, got %T", out)
	}
	payload, err := json.Marshal(e.Arguments)
	if err != nil {
		return fmt.Errorf("needle: marshal extracted arguments: %w", err)
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return fmt.Errorf("needle: decode extracted arguments into %T: %w", out, err)
	}
	return nil
}

// EngineFactory constructs an agent for a one-shot extraction. The backends'
// New functions match this signature.
type EngineFactory func(Options) (Engine, error)

// ExtractWith runs one-shot extraction on an engine built by newEngine and
// closed afterwards, mirroring the Python package's extract(). The engine is
// bound to a single tool derived from schema, so a fresh agent is created per
// call; reuse one agent and call ExtractionFromResponse for a hot loop.
func ExtractWith(newEngine EngineFactory, text string, schema any, opts Options) (*Extraction, error) {
	sc, err := SchemaFrom(schema)
	if err != nil {
		return nil, err
	}
	eopts := opts.ForExtraction(sc)
	e, err := newEngine(eopts)
	if err != nil {
		return nil, err
	}
	defer e.Close()

	resp, err := e.Complete(text)
	if err != nil {
		return nil, err
	}
	return ExtractionFromResponse(resp, sc, text, eopts.System, opts.StrictExtraction())
}

// ExtractAsWith is the typed form of ExtractWith: the schema comes from T and
// the arguments are decoded into a new T. A strict-grounding rejection still
// hands back the decoded values along with the *ExtractionError, so callers can
// use them or ignore them; no call at all gives (nil, ErrNoExtraction).
func ExtractAsWith[T any](newEngine EngineFactory, text string, opts Options) (*T, error) {
	ex, err := ExtractWith(newEngine, text, SchemaOf[T](), opts)
	if ex == nil {
		return nil, err
	}
	var v T
	if derr := ex.Decode(&v); derr != nil {
		if err != nil {
			return nil, err
		}
		return nil, derr
	}
	return &v, err
}

// ForExtraction prepares options for a one-shot extraction of schema: the
// schema becomes the only tool, the environment facts gain the local date when
// they do not carry one (relative dates such as "tomorrow" are licensed
// against it), and the defaults are filled in.
func (o Options) ForExtraction(s Schema) Options {
	o.Tools = []Tool{s.normalized().Tool()}
	if o.AutoDateFacts() {
		o.System = WithDateFact(o.System, time.Now())
	}
	return o.WithDefaults()
}

// ExtractionFromResponse turns a completed turn into an Extraction: the first
// function call, or the first suppressed one when the confidence gate withheld
// it. With strict set, values that are not grounded in text raise an
// *ExtractionError; the Extraction is still returned alongside it so the
// caller can decide what to do with the values.
func ExtractionFromResponse(resp *Response, schema Schema, text, system string, strict bool) (*Extraction, error) {
	if resp == nil {
		return nil, errors.New("needle: extraction from a nil response")
	}
	if !resp.Success {
		code := resp.ErrorCode
		if code == "" {
			code = "engine"
		}
		return nil, fmt.Errorf("needle: extract failed (%s): %s", code, resp.Error)
	}

	sc := schema.normalized()
	ex := &Extraction{
		Schema:        sc,
		Reasoning:     resp.Reasoning,
		Confidence:    resp.Confidence,
		HasConfidence: resp.HasConfidence,
		Validation:    resp.Validation,
		Raw:           resp.Raw,
		Response:      resp,
	}

	calls := resp.FunctionCalls
	if len(calls) == 0 && len(resp.SuppressedCalls) > 0 {
		calls = resp.SuppressedCalls
		ex.Suppressed = true
	}
	if len(calls) == 0 {
		return nil, ErrNoExtraction
	}
	ex.Name = calls[0].Name
	ex.Arguments = calls[0].Arguments
	if ex.Arguments == nil {
		ex.Arguments = map[string]any{}
	}

	if strict {
		if err := validateExtraction(text, sc.Parameters, ex.Arguments, ex.Validation, system); err != nil {
			return ex, err
		}
	}
	return ex, nil
}

// ExtractionError reports extracted values that are not grounded in the input,
// the Go counterpart of Python's ExtractionValidationError. Paths are
// argument-relative ("total", "items[0].price"); "negated request" marks a
// turn the engine read as a negation of the requested record.
type ExtractionError struct {
	Paths []string
}

func (e *ExtractionError) Error() string {
	paths := append([]string(nil), e.Paths...)
	sort.Strings(paths)
	return "needle: extraction returned values not grounded in the input: " + strings.Join(paths, ", ")
}
