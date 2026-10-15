package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	needle "github.com/lixquid/go-needle3"
)

// fakeEngine records the options it was constructed with and the prompt it
// was asked to complete, then returns a canned response or error.
type fakeEngine struct {
	opts   needle.Options
	prompt string

	resp *needle.Response
	err  error
}

func (f *fakeEngine) Complete(text string) (*needle.Response, error) {
	f.prompt = text
	return f.resp, f.err
}

func (f *fakeEngine) Run(query string) (*needle.Response, error) {
	return f.resp, f.err
}

func (f *fakeEngine) Reset() error { return nil }

func (f *fakeEngine) Embed(text string) ([]float32, error) { return nil, nil }

func (f *fakeEngine) Close() error { return nil }

// factory wires a prepared fakeEngine into run's engine seam.
func factory(eng *fakeEngine) func(needle.Options) (needle.Engine, error) {
	return func(opts needle.Options) (needle.Engine, error) {
		eng.opts = opts
		return eng, nil
	}
}

// noEngine is a factory that fails the test if the engine is ever created;
// used for usage-error cases that must fail before any model load.
func noEngine(t *testing.T) func(needle.Options) (needle.Engine, error) {
	t.Helper()
	return func(needle.Options) (needle.Engine, error) {
		t.Error("engine must not be created for this invocation")
		return nil, errors.New("engine must not be created")
	}
}

var sampleToolsJSON = `[
	{"name": "get_weather", "description": "Current weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
	{"name": "lights_off", "description": "Turn the lights off."}
]`

func sampleTools() []needle.Tool {
	return []needle.Tool{
		{
			Name:        "get_weather",
			Description: "Current weather.",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}},
		},
		{Name: "lights_off", Description: "Turn the lights off."},
	}
}

// sampleResponse is a representative successful turn with one call and a
// calibrated confidence.
func sampleResponse() *needle.Response {
	return &needle.Response{
		Type:          "call",
		Success:       true,
		FunctionCalls: []needle.Call{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}},
		Confidence:    0.9,
		HasConfidence: true,
		Raw:           `{"type":"call"}`,
		PrefillTPS:    100,
		DecodeTPS:     20,
	}
}

// writeToolsFile writes JSON to a temp file and returns its path.
func writeToolsFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tools.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// runCli is a convenience wrapper: run one invocation with the given argv
// (no leading command name) and piped stdin, returning the exit code, what
// was written to stdout, and what was written to stderr.
func runCli(t *testing.T, eng *fakeEngine, argv []string, stdin string) (int, string, string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	factoryFn := func(needle.Options) (needle.Engine, error) {
		return nil, errors.New("engine must not be created for this invocation")
	}
	if eng != nil {
		factoryFn = factory(eng)
	}
	code, err := run(argv, strings.NewReader(stdin), &out, &errBuf, factoryFn)
	// The returned error, when present, must have been printed to stderr
	// by main; simulate that so tests can assert on the final message.
	if err != nil {
		fmt.Fprintf(&errBuf, "needle-ask: %v\n", err)
	}
	return code, out.String(), errBuf.String()
}

func TestDefaultFacts(t *testing.T) {
	got := defaultFacts(time.Date(2026, 10, 8, 9, 41, 0, 0, time.UTC))
	want := "date: 2026-10-08 Thu 09:41"
	if got != want {
		t.Errorf("defaultFacts = %q, want %q", got, want)
	}
}

func TestLoadToolsFile(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	tools, err := loadTools(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := sampleTools()
	if len(tools) != len(want) {
		t.Fatalf("got %d tools, want %d", len(tools), len(want))
	}
	for i, w := range want {
		if tools[i].Name != w.Name || tools[i].Description != w.Description {
			t.Errorf("tools[%d] = %q/%q, want %q/%q", i, tools[i].Name, tools[i].Description, w.Name, w.Description)
		}
	}
}

func TestLoadToolsErrors(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name  string
		setup func(t *testing.T) (path string, stdin string)
		want  string // error message substring
	}{
		{
			name: "missing file",
			setup: func(t *testing.T) (string, string) {
				return filepath.Join(dir, "nope.json"), ""
			},
			want: "read tools",
		},
		{
			name: "empty file",
			setup: func(t *testing.T) (string, string) {
				return writeToolsFile(t, ""), ""
			},
			want: "empty tools input",
		},
		{
			name: "empty stdin",
			setup: func(t *testing.T) (string, string) {
				return "-", ""
			},
			want: "empty tools input",
		},
		{
			name: "invalid JSON",
			setup: func(t *testing.T) (string, string) {
				return writeToolsFile(t, "{not json"), ""
			},
			want: "parse tools JSON",
		},
		{
			name: "empty array",
			setup: func(t *testing.T) (string, string) {
				return writeToolsFile(t, "[]"), ""
			},
			want: "non-empty array",
		},
		{
			name: "missing name",
			setup: func(t *testing.T) (string, string) {
				return writeToolsFile(t, `[{"description":"x"}]`), ""
			},
			want: `tools[0]: missing "name"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path, stdin := tc.setup(t)
			_, err := loadTools(path, strings.NewReader(stdin))
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want substring %q", err, tc.want)
			}
		})
	}
}

// TestLoadToolsStdin covers the "-" path reading the toolset from the
// injected stdin.
func TestLoadToolsStdin(t *testing.T) {
	tools, err := loadTools("-", strings.NewReader(sampleToolsJSON))
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 2 || tools[0].Name != "get_weather" {
		t.Errorf("got %v, want the two sample tools", tools)
	}
}

func TestToolsJSON(t *testing.T) {
	tools := sampleTools()
	tools[0].Fn = func(map[string]any) (any, error) { return nil, nil } // must be skipped
	got := string(toolsJSON(tools))
	var parsed []map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("toolsJSON = %q: %v", got, err)
	}
	if len(parsed) != 2 {
		t.Fatalf("got %d entries, want 2", len(parsed))
	}
	if parsed[0]["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", parsed[0]["name"])
	}
	if _, ok := parsed[1]["parameters"]; !ok {
		t.Errorf("nil Parameters must render as a parameter object, got %v", parsed[1])
	}
	if _, ok := parsed[0]["fn"]; ok {
		t.Error("Fn handler must not be serialized")
	}
}

func TestReadStdin(t *testing.T) {
	if got, _ := readStdin(nil); got != "" {
		t.Errorf("readStdin(nil) = %q, want empty", got)
	}
	if got, _ := readStdin(strings.NewReader("hello\n")); got != "hello\n" {
		t.Errorf("readStdin = %q, want %q", got, "hello\n")
	}
}

func TestRunUsageErrors(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	cases := []struct {
		name   string
		argv   []string
		stdin  string
		stdout string // expected exact stdout ("" everywhere)
		stderr string // required substring on stderr
	}{
		{
			name:   "no arguments",
			argv:   []string{},
			stderr: "expected 1 or 2 arguments",
		},
		{
			name:   "too many arguments",
			argv:   []string{path, "prompt", "extra"},
			stderr: "expected 1 or 2 arguments",
		},
		{
			name:   "missing tools file",
			argv:   []string{filepath.Join(t.TempDir(), "nope.json"), "prompt"},
			stderr: "read tools",
		},
		{
			name:   "stdin tools without prompt",
			argv:   []string{"-"},
			stdin:  sampleToolsJSON,
			stderr: "prompt is mandatory when the tools are read from stdin",
		},
		{
			name:   "empty prompt file and empty stdin",
			argv:   []string{path},
			stderr: "empty prompt",
		},
		{
			name:   "unknown flag",
			argv:   []string{"-nope", path, "prompt"},
			stderr: "flag provided but not defined",
		},
		{
			name:   "json and format conflict",
			argv:   []string{"-json", "-format", "{{.Raw}}", path, "prompt"},
			stderr: "mutually exclusive",
		},
		{
			name:   "bad format template",
			argv:   []string{"-format", "{{.Tool", path, "prompt"},
			stderr: "parse format template",
		},
		{
			name:   "bad tools JSON as argument",
			argv:   []string{writeToolsFile(t, "["), "prompt"},
			stderr: "parse tools JSON",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			code, out, errOut := runCli(t, nil, tc.argv, tc.stdin)
			if code != exitUsage {
				t.Errorf("exit code = %d, want %d", code, exitUsage)
			}
			if out != tc.stdout {
				t.Errorf("stdout = %q, want %q", out, tc.stdout)
			}
			if !strings.Contains(errOut, tc.stderr) {
				t.Errorf("stderr = %q, want substring %q", errOut, tc.stderr)
			}
		})
	}
}

func TestRunHelp(t *testing.T) {
	// -h prints usage and exits 0 without touching the engine.
	code, out, errOut := runCli(t, nil, []string{"-h"}, "")
	if code != exitOK {
		t.Errorf("exit code = %d, want %d", code, exitOK)
	}
	if !strings.Contains(out+errOut, "usage: needle-ask") || !strings.Contains(out+errOut, "-confidence") {
		t.Errorf("usage not printed: stdout=%q stderr=%q", out, errOut)
	}
}

// TestRunSuccessBareTool covers the default output mode: the first tool
// call's name on stdout, exit 0.
func TestRunSuccessBareTool(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{path, "weather in Paris?"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "get_weather\n" {
		t.Errorf("stdout = %q, want %q", out, "get_weather\n")
	}
	if errOut != "" {
		t.Errorf("unexpected stderr %q", errOut)
	}
	if eng.prompt != "weather in Paris?" {
		t.Errorf("Complete prompt = %q, want the CLI prompt", eng.prompt)
	}
	if len(eng.opts.Tools) != 2 || eng.opts.Tools[0].Name != "get_weather" {
		t.Errorf("engine tools = %v, want the loaded toolset", eng.opts.Tools)
	}
}

// TestRunSuccessPromptViaStdin covers prompt piped through stdin while the
// toolset comes from a file.
func TestRunSuccessPromptViaStdin(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{path}, "what is the weather in Tokyo\n")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "get_weather\n" {
		t.Errorf("stdout = %q, want %q", out, "get_weather\n")
	}
	if eng.prompt != "what is the weather in Tokyo\n" {
		t.Errorf("Complete prompt = %q, want the piped prompt", eng.prompt)
	}
}

// TestRunSuccessToolsViaStdin covers both toolset and prompt requirements:
// tools from stdin, prompt as an argument.
func TestRunSuccessToolsViaStdin(t *testing.T) {
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{"-", "lights off please"}, sampleToolsJSON)
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "get_weather\n" {
		t.Errorf("stdout = %q, want %q", out, "get_weather\n")
	}
	if len(eng.opts.Tools) != 2 || eng.opts.Tools[1].Name != "lights_off" {
		t.Errorf("engine tools = %v, want the toolset read from stdin", eng.opts.Tools)
	}
}

func TestRunEngineOptions(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, _, errOut := runCli(t, eng, []string{"-facts", "date: 2026-10-08 Thu 09:41; locale: en-US", path, "hi"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if eng.opts.System != "date: 2026-10-08 Thu 09:41; locale: en-US" {
		t.Errorf("Options.System = %q, want the -facts value", eng.opts.System)
	}
	if len(eng.opts.Weights) == 0 {
		t.Error("Options.Weights must carry the embedded model")
	}
}

func TestRunDefaultFacts(t *testing.T) {
	// Without -facts the engine receives a synthesized "date: ..." string.
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, _, errOut := runCli(t, eng, []string{path, "hi"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if !strings.HasPrefix(eng.opts.System, "date: ") {
		t.Errorf("Options.System = %q, want the default date facts", eng.opts.System)
	}
}

func TestRunEngineLoadError(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	factory := func(needle.Options) (needle.Engine, error) { return nil, errors.New("boom") }
	var out, errBuf bytes.Buffer
	code, err := run([]string{path, "prompt"}, strings.NewReader(""), &out, &errBuf, factory)
	// main prints the returned error to stderr; simulate that here.
	if err != nil {
		fmt.Fprintf(&errBuf, "needle-ask: %v\n", err)
	}
	if code != exitFailure {
		t.Errorf("exit code = %d, want %d", code, exitFailure)
	}
	if err == nil || !strings.Contains(err.Error(), "load model") {
		t.Errorf("err = %v, want a load-model error", err)
	}
	if !strings.Contains(errBuf.String(), "needle-ask: load model") {
		t.Errorf("stderr = %q, want the load-model message", errBuf.String())
	}
	if out.String() != "" {
		t.Errorf("stdout = %q, want empty", out.String())
	}
}

func TestRunCompleteError(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: nil, err: errors.New("inference failed")}
	code, out, errOut := runCli(t, eng, []string{path, "prompt"}, "")
	if code != exitFailure {
		t.Errorf("exit code = %d, want %d", code, exitFailure)
	}
	if !strings.Contains(errOut, "complete: inference failed") {
		t.Errorf("stderr = %q, want the complete error", errOut)
	}
	if out != "" {
		t.Errorf("stdout = %q, want empty", out)
	}
}

// TestRunConfidenceGate covers the threshold gate: below it nothing is
// printed and the exit status is 1 without an error message.
func TestRunConfidenceGate(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	for _, mode := range [][]string{nil, {"-json"}, {"-format", "{{.Tool}}"}, {"-format", "{{.Confidence}}"}} {
		argv := append([]string{"-confidence", "0.5"}, mode...)
		argv = append(argv, path, "prompt")

		t.Run(strings.Join(append([]string{"mode"}, mode...), "_"), func(t *testing.T) {
			resp := sampleResponse()
			resp.Confidence = 0.4 // below the 0.5 threshold
			eng := &fakeEngine{resp: resp}
			code, out, errOut := runCli(t, eng, argv, "")
			if code != exitFailure {
				t.Errorf("exit code = %d, want %d", code, exitFailure)
			}
			if out != "" {
				t.Errorf("stdout = %q, want empty", out)
			}
			if errOut != "" {
				t.Errorf("stderr = %q, want silent exit", errOut)
			}
		})
	}
}

// TestRunConfidenceAtThreshold covers the boundary: confidence exactly at
// the threshold is reported.
func TestRunConfidenceAtThreshold(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.Confidence = 0.5
	eng := &fakeEngine{resp: resp}
	code, out, _ := runCli(t, eng, []string{"-confidence", "0.5", path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d", code, exitOK)
	}
	if out != "get_weather\n" {
		t.Errorf("stdout = %q, want the tool name", out)
	}
}

// TestRunNoConfidenceHead covers weights without a calibration head: the
// gate must not suppress the output regardless of threshold.
func TestRunNoConfidenceHead(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.HasConfidence = false
	eng := &fakeEngine{resp: resp}
	code, out, errOut := runCli(t, eng, []string{"-confidence", "0.9", path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "get_weather\n" {
		t.Errorf("stdout = %q, want the tool name", out)
	}
}

// TestRunOffTopic covers the empty-call refusal: no calls, no match, but a
// successful turn still exits 0 (the confidence gate alone decides silence).
func TestRunOffTopic(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.FunctionCalls = nil
	resp.Type = "call" // off-topic refusal: type call with no calls
	eng := &fakeEngine{resp: resp}
	code, out, errOut := runCli(t, eng, []string{path, "tell me a joke"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	// An empty rendered tool name stays empty: no trailing newline is added.
	if out != "" {
		t.Errorf("stdout = %q, want empty", out)
	}
}

func TestRunJSONOutput(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{"-json", path, "weather in Paris?"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("stdout is not JSON: %v (%q)", err, out)
	}
	eq := func(key string, want any) {
		t.Helper()
		if got := doc[key]; fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("doc[%q] = %v, want %v", key, got, want)
		}
	}
	eq("type", "call")
	eq("success", true)
	eq("matched", true)
	eq("matched_tool", "get_weather")
	eq("above_threshold", true)
	eq("prompt", "weather in Paris?")
	eq("tools_path", path)
	eq("facts_source", "default")
	if f, _ := doc["facts"].(string); !strings.HasPrefix(f, "date: ") {
		t.Errorf("doc[\"facts\"] = %v, want default date facts", doc["facts"])
	}
	conf := doc["confidence"].(map[string]any)
	if conf["value"].(float64) != 0.9 || conf["available"] != true || conf["threshold"].(float64) != 0 {
		t.Errorf("confidence block = %v", conf)
	}
	calls := doc["function_calls"].([]any)
	if len(calls) != 1 || calls[0].(map[string]any)["name"] != "get_weather" {
		t.Errorf("function_calls = %v", calls)
	}
	// The serialized toolset must match the engine's canonical form.
	wantTools := string(toolsJSON(sampleTools()))
	if gotTools, _ := json.Marshal(doc["tools"]); string(gotTools) != wantTools {
		t.Errorf("tools = %s, want %s", gotTools, wantTools)
	}
}

func TestRunJSONOutputWithFactsFlag(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{"-json", "-facts", "date: 2026-10-08 Thu 09:41; device: laptop", path, "hi"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("stdout is not JSON: %v", err)
	}
	if doc["facts"] != "date: 2026-10-08 Thu 09:41; device: laptop" || doc["facts_source"] != "flag" {
		t.Errorf("facts = %v (source %v), want flag-provided facts", doc["facts"], doc["facts_source"])
	}
}

func TestRunJSONOutputErrorFields(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.Error = "engine blew up"
	resp.ErrorCode = "E_INTERNAL"
	eng := &fakeEngine{resp: resp}
	code, out, errOut := runCli(t, eng, []string{"-json", path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("stdout is not JSON: %v", err)
	}
	if doc["error"] != "engine blew up" || doc["error_code"] != "E_INTERNAL" {
		t.Errorf("error fields = %v / %v", doc["error"], doc["error_code"])
	}
}

func TestRunFormat(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	cases := []struct {
		name   string
		format string
		want   string
	}{
		{"bare tool field", "{{.Tool}}", "get_weather\n"},
		{"args access", "{{index .Args \"city\"}}", "Paris\n"},
		{"confidence", "conf={{.Confidence}}", "conf=0.9\n"},
		{"success", "ok={{.Success}}", "ok=true\n"},
		{"no trailing newline in template", "{{.Tool}}!", "get_weather!\n"},
		{"multiline keeps trailing newline", "{{.Tool}}\n--\n", "get_weather\n--\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			eng := &fakeEngine{resp: sampleResponse()}
			code, out, errOut := runCli(t, eng, []string{"-format", tc.format, path, "prompt"}, "")
			if code != exitOK {
				t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
			}
			if out != tc.want {
				t.Errorf("stdout = %q, want %q", out, tc.want)
			}
		})
	}
}

// TestRunFormatRawField checks that .Raw exposes the full -json object.
func TestRunFormatRawField(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{"-format", `{{.Raw.matched_tool}}/{{.Raw.confidence.value}}/{{.Raw.facts_source}}`, path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "get_weather/0.9/default\n" {
		t.Errorf("stdout = %q, want %q", out, "get_weather/0.9/default\n")
	}
}

// TestRunFormatExecuteError covers a template that parses fine but fails at
// execution time: that is an internal failure (exit 1), not a usage error.
func TestRunFormatExecuteError(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	eng := &fakeEngine{resp: sampleResponse()}
	code, out, errOut := runCli(t, eng, []string{"-format", "{{.Success.Nope}}", path, "prompt"}, "")
	if code != exitFailure {
		t.Errorf("exit code = %d, want %d", code, exitFailure)
	}
	if !strings.Contains(errOut, "execute format template") {
		t.Errorf("stderr = %q, want the execute error", errOut)
	}
	if out != "" {
		t.Errorf("stdout = %q, want empty", out)
	}
}

// TestRunFormatNilArguments covers a call without an Arguments map: .Args
// must be an (empty) map, not <no value>.
func TestRunFormatNilArguments(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.FunctionCalls[0].Arguments = nil
	eng := &fakeEngine{resp: resp}
	code, out, errOut := runCli(t, eng, []string{"-format", "args={{len .Args}}", path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	if out != "args=0\n" {
		t.Errorf("stdout = %q, want %q", out, "args=0\n")
	}
}

// TestRunSuppressedCalls covers a turn where the engine withheld the call.
func TestRunSuppressedCalls(t *testing.T) {
	path := writeToolsFile(t, sampleToolsJSON)
	resp := sampleResponse()
	resp.FunctionCalls = nil
	resp.SuppressedCalls = []needle.Call{{Name: "lights_off"}}
	eng := &fakeEngine{resp: resp}
	code, out, errOut := runCli(t, eng, []string{"-json", path, "prompt"}, "")
	if code != exitOK {
		t.Fatalf("exit code = %d, want %d (stderr %q)", code, exitOK, errOut)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("stdout is not JSON: %v", err)
	}
	eq := func(key string, want any) {
		t.Helper()
		if got := doc[key]; fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("doc[%q] = %v, want %v", key, got, want)
		}
	}
	eq("matched", false) // no function calls -> not matched
	eq("matched_tool", nil)
	eq("above_threshold", true)
	suppressed := doc["suppressed_calls"].([]any)
	if len(suppressed) != 1 || suppressed[0].(map[string]any)["name"] != "lights_off" {
		t.Errorf("suppressed_calls = %v", suppressed)
	}
}
