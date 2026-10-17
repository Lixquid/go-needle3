package needle

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

type lineItem struct {
	Price float64 `json:"price" description:"unit price"`
}

type invoice struct {
	Vendor   string     `json:"vendor" description:"who issued it"`
	DueDate  Date       `json:"due_date" description:"when it is due"`
	Total    float64    `json:"total,omitempty"`
	Note     *string    `json:"note,omitempty"`
	Priority string     `json:"priority,omitempty" enum:"low,normal,high"`
	Items    []lineItem `json:"items,omitempty"`
	Secret   string     `json:"-"`
}

func (invoice) Description() string { return "An invoice to pay." }

func TestSchemaOf(t *testing.T) {
	sc := SchemaOf[invoice]()
	if sc.Name != "invoice" {
		t.Errorf("name = %q, want the type name", sc.Name)
	}
	if sc.Description != "An invoice to pay." {
		t.Errorf("description = %q", sc.Description)
	}
	if got, _ := sc.Parameters["type"].(string); got != "object" {
		t.Fatalf("parameters type = %v", sc.Parameters["type"])
	}
	props, _ := sc.Parameters["properties"].(map[string]any)
	for _, missing := range []string{"vendor", "due_date", "total", "note", "priority", "items"} {
		if _, ok := props[missing]; !ok {
			t.Errorf("properties missing %q: %v", missing, props)
		}
	}
	if _, leaked := props["Secret"]; leaked {
		t.Error(`json:"-" field leaked into the schema`)
	}

	due, _ := props["due_date"].(map[string]any)
	if due["format"] != "date" {
		t.Errorf("due_date format = %v, want date (Date must mark itself)", due["format"])
	}
	prio, _ := props["priority"].(map[string]any)
	if want := []string{"low", "normal", "high"}; !reflect.DeepEqual(prio["enum"], want) {
		t.Errorf("priority enum = %#v, want %v", prio["enum"], want)
	}
	item, _ := props["items"].(map[string]any)
	if item["type"] != "array" {
		t.Fatalf("items type = %v, want array", item["type"])
	}
	elem, _ := item["items"].(map[string]any)
	elemProps, _ := elem["properties"].(map[string]any)
	if _, ok := elemProps["price"]; !ok {
		t.Errorf("nested item properties = %v", elem["properties"])
	}

	// Only vendor and due_date are required: a pointer, omitempty or a default
	// makes a field optional.
	req, _ := sc.Parameters["required"].([]string)
	if !reflect.DeepEqual(req, []string{"vendor", "due_date"}) {
		t.Errorf("required = %v, want [vendor due_date]", req)
	}

	// The whole schema must render as one engine tool entry.
	tools, err := Options{Tools: []Tool{sc.Tool()}}.ToolsJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(tools), `"name":"invoice"`) || !strings.Contains(string(tools), `"due_date"`) {
		t.Errorf("tools JSON = %s", tools)
	}
}

type recursive struct {
	Name string     `json:"name"`
	Next *recursive `json:"next,omitempty"`
}

func TestSchemaOfRecursive(t *testing.T) {
	if _, err := json.Marshal(SchemaOf[recursive]().Parameters); err != nil {
		t.Fatal(err)
	}
	sc := SchemaOf[map[string]any]()
	if sc.Name != DefaultSchemaName {
		t.Errorf("name = %q, want %q for an anonymous type", sc.Name, DefaultSchemaName)
	}
	if got, _ := SchemaOf[string]().Parameters["type"].(string); got != "object" {
		t.Errorf("a scalar type must still yield an object parameter, got %v", SchemaOf[string]().Parameters)
	}
}

func TestSchemaFrom(t *testing.T) {
	params := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}

	t.Run("tool entry map", func(t *testing.T) {
		sc, err := SchemaFrom(map[string]any{"name": "Weather", "description": "d", "parameters": params})
		if err != nil {
			t.Fatal(err)
		}
		if sc.Name != "Weather" || sc.Description != "d" {
			t.Errorf("schema = %+v", sc)
		}
	})
	t.Run("bare parameters", func(t *testing.T) {
		sc, err := SchemaFrom(params)
		if err != nil {
			t.Fatal(err)
		}
		if sc.Name != DefaultSchemaName {
			t.Errorf("name = %q, want the default", sc.Name)
		}
		if sc.Parameters["properties"] == nil {
			t.Error("parameters lost")
		}
	})
	t.Run("missing type gets object", func(t *testing.T) {
		sc, err := SchemaFrom(map[string]any{"properties": map[string]any{}})
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := sc.Parameters["type"].(string); got != "object" {
			t.Errorf("type = %v, want object", sc.Parameters["type"])
		}
	})
	t.Run("json text", func(t *testing.T) {
		sc, err := SchemaFrom(`{"name":"Contact","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}`)
		if err != nil {
			t.Fatal(err)
		}
		if sc.Name != "Contact" {
			t.Errorf("name = %q", sc.Name)
		}
		if _, err := SchemaFrom("{oops"); err == nil {
			t.Error("bad JSON should error")
		}
	})
	t.Run("struct value and type", func(t *testing.T) {
		sc, err := SchemaFrom(invoice{})
		if err != nil {
			t.Fatal(err)
		}
		if sc.Name != "invoice" {
			t.Errorf("name = %q", sc.Name)
		}
		sc2, err := SchemaFrom(reflect.TypeOf(lineItem{}))
		if err != nil {
			t.Fatal(err)
		}
		if sc2.Name != "lineItem" {
			t.Errorf("name = %q", sc2.Name)
		}
	})
	t.Run("tool and schema", func(t *testing.T) {
		if sc, err := SchemaFrom(Tool{Name: "T", Parameters: params}); err != nil || sc.Name != "T" {
			t.Errorf("Tool form: %+v %v", sc, err)
		}
		if sc, err := SchemaFrom(Schema{}); err != nil || sc.Name != DefaultSchemaName || sc.Parameters == nil {
			t.Errorf("empty Schema: %+v %v", sc, err)
		}
	})
	t.Run("rejects nonsense", func(t *testing.T) {
		if _, err := SchemaFrom(nil); err == nil {
			t.Error("nil should error")
		}
		if _, err := SchemaFrom(42); err == nil {
			t.Error("an int is not a schema")
		}
	})
}

func TestDateJSON(t *testing.T) {
	var d Date
	if err := json.Unmarshal([]byte(`"2026-09-01"`), &d); err != nil {
		t.Fatal(err)
	}
	if d.String() != "2026-09-01" {
		t.Errorf("date = %s", d)
	}
	if got := d.Time().Format("2006-01-02"); got != "2026-09-01" {
		t.Errorf("Time() = %s", got)
	}
	if err := json.Unmarshal([]byte(`"2026-09-01T14:30:00Z"`), &d); err != nil {
		t.Fatalf("RFC 3339 should parse: %v", err)
	}
	if err := json.Unmarshal([]byte(`""`), &d); err != nil {
		t.Fatalf("empty string should give the zero date: %v", err)
	}
	if err := json.Unmarshal([]byte(`"soon"`), &d); err == nil {
		t.Error("garbage should fail")
	}
	out, err := json.Marshal(struct {
		D Date `json:"d"`
	}{Date(time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC))})
	if err != nil || string(out) != `{"d":"2026-09-01"}` {
		t.Errorf("marshal = %s %v", out, err)
	}
}

func TestExtractionFromResponse(t *testing.T) {
	sc := Schema{Name: "Contact", Parameters: map[string]any{"type": "object"}}
	dateSchema := Schema{Name: "Contact", Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"due_date": map[string]any{"type": "string", "format": "date"},
		},
	}}

	// A plain call.
	resp := &Response{Type: "call", Success: true, Confidence: 0.8, HasConfidence: true,
		FunctionCalls: []Call{{Name: "Contact", Arguments: map[string]any{"name": "John"}}}}
	ex, err := ExtractionFromResponse(resp, sc, "John at john@doe.com", "", true)
	if err != nil {
		t.Fatal(err)
	}
	if ex.Name != "Contact" || ex.Suppressed || ex.Confidence != 0.8 {
		t.Errorf("extraction = %+v", ex)
	}
	var out struct {
		Name string `json:"name"`
	}
	if err := ex.Decode(&out); err != nil || out.Name != "John" {
		t.Errorf("decode = %+v %v", out, err)
	}

	// A withheld call still yields its values, flagged as suppressed.
	resp = &Response{Type: "call", Success: true, Confidence: 0.04,
		SuppressedCalls: []Call{{Name: "Contact", Arguments: map[string]any{"name": "John"}}}}
	ex, err = ExtractionFromResponse(resp, sc, "text", "", true)
	if err != nil || !ex.Suppressed {
		t.Errorf("suppressed extraction = %+v %v", ex, err)
	}

	// No call at all.
	if _, err := ExtractionFromResponse(&Response{Type: "respond", Success: true}, sc, "t", "", true); !errors.Is(err, ErrNoExtraction) {
		t.Errorf("off-topic = %v, want ErrNoExtraction", err)
	}
	// An engine failure surfaces as an error, not an empty record.
	if _, err := ExtractionFromResponse(&Response{Success: false, Error: "boom", ErrorCode: "parse"}, sc, "t", "", true); err == nil {
		t.Error("a failed turn must error")
	}
	// strict=false lets an ungrounded year through, and the extraction is
	// still handed back alongside the error.
	resp = &Response{Type: "call", Success: true, FunctionCalls: []Call{
		{Name: "Contact", Arguments: map[string]any{"due_date": "1999-01-02"}}}}
	ex, err = ExtractionFromResponse(resp, dateSchema, "due on 5th September 2031", "", true)
	var exErr *ExtractionError
	if !errors.As(err, &exErr) || ex == nil {
		t.Errorf("strict should reject the fabricated year and still return the values: %+v %v", ex, err)
	}
	if ex, err := ExtractionFromResponse(resp, dateSchema, "due on 5th September 2031", "", false); err != nil || ex == nil {
		t.Errorf("permissive: %+v %v", ex, err)
	}
}

func TestDecodeTargets(t *testing.T) {
	ex := &Extraction{Arguments: map[string]any{"a": 1.0, "b": "x"}}
	var v struct {
		A int    `json:"a"`
		B string `json:"b"`
	}
	if err := ex.Decode(&v); err != nil || v.A != 1 || v.B != "x" {
		t.Errorf("decode = %+v %v", v, err)
	}
	if err := ex.Decode(v); err == nil {
		t.Error("a non-pointer target must error")
	}
	if err := (&Extraction{}).Decode(&v); !errors.Is(err, ErrNoExtraction) {
		t.Errorf("no arguments = %v, want ErrNoExtraction", err)
	}
}

func TestForExtraction(t *testing.T) {
	opts := Options{Weights: []byte{1, 2, 3}}
	sc := Schema{Name: "Invoice", Parameters: map[string]any{"type": "object"}}
	got := opts.ForExtraction(sc)
	if len(got.Tools) != 1 || got.Tools[0].Name != "Invoice" {
		t.Errorf("tools = %+v", got.Tools)
	}
	if !strings.HasPrefix(got.System, "date: ") {
		t.Errorf("system = %q, want the auto date fact", got.System)
	}
	if got.BufferSize != DefaultBufferSize || got.MaxNewTokens != DefaultMaxNewTokens {
		t.Errorf("defaults not applied: %+v", got)
	}
	off := Options{AutoDate: Bool(false)}
	if off.ForExtraction(sc).System != "" {
		t.Error("AutoDate=false must leave the system prompt alone")
	}
	if !opts.StrictExtraction() {
		t.Error("strict must default on")
	}
	o2 := Options{Strict: Bool(false)}
	if o2.StrictExtraction() {
		t.Error("Strict=false must disable grounding")
	}
}

// level and cursor both describe their own schema; one uses a value receiver
// and the other a pointer receiver, which reflection must also accept.
type level string

func (level) JSONSchema() map[string]any {
	return map[string]any{"type": "string", "enum": []string{"low", "high"}}
}

type cursor struct{ X, Y int }

func (*cursor) JSONSchema() map[string]any {
	return map[string]any{"type": "array", "items": map[string]any{"type": "integer"}, "minItems": 2}
}

func TestSchemaProvider(t *testing.T) {
	type pair struct {
		L level  `json:"l"`
		C cursor `json:"c"`
	}
	sc, err := SchemaFrom(pair{})
	if err != nil {
		t.Fatal(err)
	}
	props, _ := sc.Parameters["properties"].(map[string]any)

	l, _ := props["l"].(map[string]any)
	if l["enum"] == nil {
		t.Errorf("value-receiver JSONSchema ignored: %v", l)
	}
	c, _ := props["c"].(map[string]any)
	if c["type"] != "array" || c["minItems"] != 2 {
		t.Errorf("pointer-receiver JSONSchema ignored: %v", c)
	}
}

func TestReflectionTagsAndKinds(t *testing.T) {
	type embedded struct {
		Room string `json:"room" description:"room name"`
	}
	type record struct {
		embedded
		Count    int            `json:"count" minimum:"0" maximum:"100"`
		Rate     float64        `json:"rate" minimum:"0.5"`
		Skip     string         `json:"-"`
		Raw      []byte         `json:"raw,omitempty"`
		Labels   map[string]int `json:"labels,omitempty"`
		Stamp    time.Time      `json:"stamp,omitempty"`
		Anything any            `json:"anything,omitempty"`
		Forced   *string        `json:"forced" required:"true"`
		Nixed    string         `json:"nixed" required:"false"`
		Has      string         `json:"has" default:"auto"`
	}
	sc := SchemaOf[record]()
	props := sc.Parameters["properties"].(map[string]any)

	if _, ok := props["Skip"]; ok {
		t.Error(`json:"-" must be dropped`)
	}
	if _, ok := props["room"]; !ok {
		t.Error("an unnamed embedded struct must be inlined")
	}
	count := props["count"].(map[string]any)
	if count["minimum"] != int64(0) || count["maximum"] != int64(100) {
		t.Errorf("integer bounds should be ints: %v", count)
	}
	rate := props["rate"].(map[string]any)
	if rate["minimum"] != 0.5 {
		t.Errorf("float bounds: %v", rate)
	}
	if props["raw"].(map[string]any)["type"] != "string" {
		t.Error("[]byte travels as a JSON string")
	}
	if props["stamp"].(map[string]any)["format"] != "date-time" {
		t.Error("time.Time should map to format date-time")
	}
	// An interface has no static shape, so it falls back to free text, which
	// is what the reference does for an unannotated field.
	if got, _ := props["anything"].(map[string]any)["type"].(string); got != "string" {
		t.Errorf("an interface field should be free text: %v", props["anything"])
	}
	if props["has"].(map[string]any)["default"] != "auto" {
		t.Error("a default tag should reach the schema")
	}

	required, _ := sc.Parameters["required"].([]string)
	got := map[string]bool{}
	for _, r := range required {
		got[r] = true
	}
	if !got["forced"] {
		t.Error(`required:"true" must force a pointer field in`)
	}
	if got["nixed"] || got["has"] {
		t.Errorf("required set = %v (nixed/has must not be required)", required)
	}
	if !got["room"] {
		t.Error("an inlined embedded field follows the same required rule as any other")
	}
}

// stubEngine stands in for a backend so the shared Extract plumbing can be
// tested without a model: it records the Options it was constructed with and
// replays one canned turn.
type stubEngine struct {
	opts   Options
	text   string
	resp   *Response
	err    error
	closed bool
}

func (s *stubEngine) Complete(text string) (*Response, error) {
	s.text = text
	return s.resp, s.err
}
func (s *stubEngine) Run(q string) (*Response, error) { return s.Complete(q) }
func (s *stubEngine) Reset() error                    { return nil }
func (s *stubEngine) Embed(string) ([]float32, error) { return nil, nil }
func (s *stubEngine) Close() error                    { s.closed = true; return nil }

func stubFactory(resp *Response) (func(Options) (Engine, error), *stubEngine) {
	var held stubEngine
	return func(o Options) (Engine, error) {
		held = stubEngine{opts: o, resp: resp}
		return &held, nil
	}, &held
}

func TestExtractWithPlumbing(t *testing.T) {
	resp := &Response{Type: "call", Success: true, Confidence: 0.7, FunctionCalls: []Call{
		{Name: "Invoice", Arguments: map[string]any{"due_date": "2031-09-05"}}}}
	newEngine, held := stubFactory(resp)

	sc := Schema{Name: "Invoice", Parameters: extractSchema}
	ex, err := ExtractWith(newEngine, "due on 5th September 2031", sc, Options{Weights: []byte{1}})
	if err != nil {
		t.Fatal(err)
	}
	if ex.Name != "Invoice" || ex.Confidence != 0.7 {
		t.Errorf("extraction = %+v", ex)
	}
	// The schema must reach the engine as the only tool, the text as the turn,
	// and the agent must be closed again.
	if len(held.opts.Tools) != 1 || held.opts.Tools[0].Name != "Invoice" {
		t.Errorf("bound tools = %+v", held.opts.Tools)
	}
	if !strings.HasPrefix(held.opts.System, "date: ") {
		t.Errorf("auto date fact missing: %q", held.opts.System)
	}
	if held.opts.BufferSize != DefaultBufferSize || held.opts.MaxNewTokens != DefaultMaxNewTokens {
		t.Errorf("defaults not applied: %+v", held.opts)
	}
	if held.text != "due on 5th September 2031" {
		t.Errorf("turn text = %q", held.text)
	}
	if !held.closed {
		t.Error("the agent was not closed")
	}
}

func TestExtractWithStrictnessAndFailures(t *testing.T) {
	dateSchema := Schema{Name: "Invoice", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"due_date": map[string]any{"type": "string", "format": "date"}},
	}}
	resp := &Response{Type: "call", Success: true, FunctionCalls: []Call{
		{Name: "Invoice", Arguments: map[string]any{"due_date": "1999-01-02"}}}}

	// Strict: grounded in nothing, so an *ExtractionError with values attached.
	newEngine, _ := stubFactory(resp)
	ex, err := ExtractWith(newEngine, "due on 5th September 2031", dateSchema, Options{Weights: []byte{1}})
	var exErr *ExtractionError
	if !errors.As(err, &exErr) || ex == nil {
		t.Fatalf("want a grounding error with values, got %+v %v", ex, err)
	}
	// Permissive: the same turn comes back clean.
	newEngine, _ = stubFactory(resp)
	if _, err := ExtractWith(newEngine, "due on 5th September 2031", dateSchema,
		Options{Weights: []byte{1}, Strict: Bool(false)}); err != nil {
		t.Errorf("permissive: %v", err)
	}
	// AutoDate=false leaves the facts alone, so nothing licenses a relative year.
	newEngine, held := stubFactory(resp)
	if _, err := ExtractWith(newEngine, "due tomorrow", dateSchema,
		Options{Weights: []byte{1}, AutoDate: Bool(false)}); err != nil {
		t.Errorf("permissive auto-date: %v", err)
	}
	if held.opts.System != "" {
		t.Errorf("AutoDate=false changed the system prompt to %q", held.opts.System)
	}
	// No call at all.
	newEngine, _ = stubFactory(&Response{Type: "respond", Success: true})
	if _, err := ExtractWith(newEngine, "off topic", dateSchema, Options{Weights: []byte{1}}); !errors.Is(err, ErrNoExtraction) {
		t.Errorf("want ErrNoExtraction, got %v", err)
	}
	// A factory failure propagates and is not wrapped in an extraction.
	if _, err := ExtractWith(func(Options) (Engine, error) { return nil, errors.New("no slot") },
		"text", dateSchema, Options{Weights: []byte{1}}); err == nil || !strings.Contains(err.Error(), "no slot") {
		t.Errorf("factory error = %v", err)
	}
	// A bad schema never reaches the engine.
	if _, err := ExtractWith(func(Options) (Engine, error) { return &stubEngine{}, nil },
		"text", 42, Options{Weights: []byte{1}}); err == nil {
		t.Error("a bad schema must error before the engine is built")
	}
}

func TestExtractAsWith(t *testing.T) {
	resp := &Response{Type: "call", Success: true, FunctionCalls: []Call{
		{Name: "invoice", Arguments: map[string]any{"vendor": "Acme", "total": 1200.0}}}}

	newEngine, _ := stubFactory(resp)
	got, err := ExtractAsWith[invoice](newEngine, "Invoice from Acme for $1,200", Options{Weights: []byte{1}})
	if err != nil {
		t.Fatal(err)
	}
	if got.Vendor != "Acme" || got.Total != 1200 {
		t.Errorf("decoded = %+v", got)
	}

	// A strict rejection still returns the decoded record with the error.
	resp = &Response{Type: "call", Success: true, FunctionCalls: []Call{
		{Name: "invoice", Arguments: map[string]any{"vendor": "Acme", "due_date": "1999-01-02"}}}}
	newEngine, _ = stubFactory(resp)
	got, err = ExtractAsWith[invoice](newEngine, "due on 5th September 2031", Options{Weights: []byte{1}})
	var exErr *ExtractionError
	if !errors.As(err, &exErr) {
		t.Fatalf("want an *ExtractionError, got %v", err)
	}
	if got == nil || got.Vendor != "Acme" {
		t.Errorf("values should survive the rejection, got %+v", got)
	}

	// No call: nil and ErrNoExtraction.
	newEngine, _ = stubFactory(&Response{Type: "respond", Success: true})
	if got, err := ExtractAsWith[invoice](newEngine, "off topic", Options{Weights: []byte{1}}); !errors.Is(err, ErrNoExtraction) || got != nil {
		t.Errorf("no call = %+v %v", got, err)
	}
}
