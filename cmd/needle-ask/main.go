// Command needle-ask is a one-shot Needle 3 tool matcher built on the cgo
// backend. Given a JSON toolset and a prompt, it prints the tool the model
// would call.
//
// Usage:
//
//	needle-ask [flags] <tools.json|-> [prompt]
//
// The tools path is required; pass "-" to read the tools JSON from stdin,
// in which case the prompt must be given as an argument. Otherwise the
// prompt may be given as an argument or piped via stdin.
//
// Flags:
//
//		-facts string
//		    Environment facts passed to the model as its system context.
//	   Format: colon-delimited key/value pairs separated by semicolons,
//		    e.g. "date: 2026-10-08 Thu 09:41; locale: en-US; device: laptop".
//		    Defaults to "date: <current local datetime>".
//		-confidence float
//		    Minimum confidence (0..1) required to report a match. Default 0
//		    (always report). Below the threshold, nothing is printed and
//		    the exit status is 1.
//		-json
//		    Instead of the bare tool name, emit one JSON object with the full
//		    response: matched tool(s) and arguments, confidence vs. threshold,
//		    suppressed calls, reasoning, the effective facts, the prompt and
//		    the loaded toolset.
//		-format string
//		    Render the output with a Go text/template instead of the bare
//		    tool name. The template receives:
//		      .Tool       name of the first function call ("" if none)
//		      .Args       arguments map of the first function call
//		      .Confidence the 0..1 confidence of the match
//		      .Success    whether the turn succeeded
//		      .Raw        the full data object the -json flag would emit
//		    A trailing newline is added if the rendered output lacks one.
//		    Defaults to "{{.Tool}}" (the first tool call's name).
//		    Mutually exclusive with -json.
//
// Exit status:
//
//	0   a match was reported
//	1   the result did not meet the -confidence threshold (no output),
//	    or an internal error such as model load or inference failure
//	    (message on stderr)
//	2   user error: bad flags, bad tools JSON, bad -format template, ...
//	    (message on stderr)
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/template"
	"time"

	needle "github.com/lixquid/go-needle3"
	"github.com/lixquid/go-needle3/engine/cgo"
	"github.com/lixquid/go-needle3/model"
)

const usage = `usage: needle-ask [flags] <tools.json|-> [prompt]

Matches a prompt against a toolset with the Needle 3 tool-calling model.

Arguments:
  tools.json   path to a JSON file with the toolset:
               [{"name":...,"description":...,"parameters":{...}}, ...]
               "-" reads the toolset JSON from stdin (prompt must then be
               an argument, since stdin carries the tools).
  prompt       the user query. Optional: if omitted, it is read from stdin.

Flags:`

func main() {
	// nil stdin means an interactive terminal: nothing is being piped, so
	// run must not block trying to read it.
	var stdin io.Reader
	if st, err := os.Stdin.Stat(); err != nil || st.Mode()&os.ModeCharDevice == 0 {
		stdin = os.Stdin // pipe or file: run may consume it
	}
	code, err := run(os.Args[1:], stdin, os.Stdout, os.Stderr, newEngine)
	if err != nil {
		fmt.Fprintln(os.Stderr, "needle-ask:", err)
	}
	os.Exit(code)
}

// newEngine builds the cgo backend; a seam so tests can inject a fake
// engine instead of loading the real model.
func newEngine(opts needle.Options) (needle.Engine, error) {
	return cgo.New(opts)
}

// Exit codes.
const (
	exitOK      = 0 // match reported
	exitFailure = 1 // below threshold (silent) or internal error
	exitUsage   = 2 // user error: bad flags, tools JSON, template, ...
)

// run performs one invocation and returns the process exit code plus an
// error to print on stderr (nil for the silent below-threshold exit).
// run performs one invocation and returns the process exit code plus an
// error to print on stderr (nil for the silent below-threshold exit). argv,
// stdin, stdout and stderr are injectable for tests; newEngine stands in
// for the backend constructor.
func run(argv []string, stdin io.Reader, stdout, stderr io.Writer, newEngine func(needle.Options) (needle.Engine, error)) (int, error) {
	cfg := config{}
	fs := flag.NewFlagSet("needle-ask", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cfg.facts, "facts", "", `environment facts for the model, as colon-delimited key/value pairs separated by semicolons, e.g. "date: 2026-10-08 Thu 09:41; locale: en-US". Default: "date: <current local datetime>"`)
	fs.Float64Var(&cfg.threshold, "confidence", 0, "minimum confidence (0..1) required to report a match; 0 always reports")
	fs.BoolVar(&cfg.jsonOut, "json", false, "emit a JSON object with the full response instead of the bare tool name")
	fs.StringVar(&cfg.format, "format", "{{.Tool}}", `go text/template rendered instead of the bare tool name; fields: .Tool (first call name), .Args (its arguments map), .Confidence (0..1), .Success (bool), .Raw (the full -json data object)`)
	fs.Usage = func() {
		fmt.Fprintln(stderr, usage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) { // -h/-help: usage already printed
			return exitOK, nil
		}
		return exitUsage, err
	}

	args := fs.Args()
	if len(args) < 1 || len(args) > 2 {
		fs.Usage()
		return exitUsage, errors.New("expected 1 or 2 arguments")
	}
	toolsPath, prompt := args[0], ""
	if len(args) == 2 {
		prompt = args[1]
	}

	tools, err := loadTools(toolsPath, stdin)
	if err != nil {
		return exitUsage, err
	}
	if toolsPath == "-" && prompt == "" {
		fs.Usage()
		return exitUsage, errors.New("prompt is mandatory when the tools are read from stdin (-)")
	}
	if prompt == "" {
		prompt, err = readStdin(stdin)
		if err != nil {
			return exitUsage, err
		}
		if prompt == "" {
			return exitUsage, errors.New("empty prompt")
		}
	}

	// -json overrides the template only when -format was asked for
	// explicitly; the default value is not treated as a conflict.
	formatSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "format" {
			formatSet = true
		}
	})
	if formatSet && cfg.jsonOut {
		return exitUsage, errors.New("flags -json and -format are mutually exclusive")
	}

	// Validate the -format template up front, before the model is loaded:
	// a parse failure is a user error and should fail fast.
	tmpl, err := template.New("output").Parse(cfg.format)
	if err != nil {
		return exitUsage, fmt.Errorf("parse format template: %w", err)
	}
	cfg.tmpl = tmpl

	facts, factsSource := cfg.facts, "flag"
	if facts == "" {
		facts = defaultFacts(time.Now())
		factsSource = "default"
	}

	agent, err := newEngine(needle.Options{
		Tools:   tools,
		System:  facts,
		Weights: model.Needle3,
	})
	if err != nil {
		return exitFailure, fmt.Errorf("load model: %w", err)
	}
	defer agent.Close()

	resp, err := agent.Complete(prompt)
	if err != nil {
		return exitFailure, fmt.Errorf("complete: %w", err)
	}

	names := make([]string, len(resp.FunctionCalls))
	for i, c := range resp.FunctionCalls {
		names[i] = c.Name
	}
	above := !resp.HasConfidence || resp.Confidence >= cfg.threshold

	// The confidence gate applies to every output mode: below the
	// threshold nothing is printed anywhere and the exit status is 1
	// (no error, so main stays silent).
	if !above {
		return exitFailure, nil
	}

	// JSON mode: emit the full response object. Otherwise render the
	// -format template (default "{{.Tool}}" = the first tool call's name)
	// against the response data.
	var emitErr error
	if cfg.jsonOut {
		emitErr = emitJSON(stdout, resp, names, above, facts, factsSource, prompt, toolsPath, tools, cfg.threshold)
	} else {
		emitErr = emitFormat(stdout, resp, names, above, facts, factsSource, prompt, toolsPath, tools, cfg.threshold, cfg.tmpl)
	}
	if emitErr != nil {
		return exitFailure, emitErr
	}
	return exitOK, nil
}

type config struct {
	facts     string
	threshold float64
	jsonOut   bool
	format    string
	tmpl      *template.Template
}

// defaultFacts renders the fallback system facts, e.g.
// "date: 2026-10-08 Thu 09:41".
func defaultFacts(t time.Time) string {
	return "date: " + t.Format("2006-01-02 Mon 15:04")
}

// loadTools reads the toolset JSON from a file, or from stdin when path is
// "-". The accepted shape is the engine's own:
//
//	[{"name":...,"description":...,"parameters":{...}}]
func loadTools(path string, stdin io.Reader) ([]needle.Tool, error) {
	var (
		data []byte
		err  error
	)
	if path == "-" {
		var s string
		if s, err = readStdin(stdin); err != nil {
			return nil, err
		}
		data = []byte(s)
	} else if data, err = os.ReadFile(path); err != nil {
		return nil, fmt.Errorf("read tools: %w", err)
	}
	if len(data) == 0 {
		return nil, errors.New("empty tools input")
	}
	var tools []needle.Tool
	if err := json.Unmarshal(data, &tools); err != nil {
		return nil, fmt.Errorf("parse tools JSON: %w", err)
	}
	if len(tools) == 0 {
		return nil, errors.New("tools JSON must be a non-empty array")
	}
	for i, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("tools[%d]: missing \"name\"", i)
		}
	}
	return tools, nil
}

// toolsJSON renders the toolset into the engine's canonical JSON array
// ([{"name":...,"description":...,"parameters":{...}}]), skipping the
// non-serializable Tool.Fn handler field.
func toolsJSON(tools []needle.Tool) json.RawMessage {
	data, err := needle.Options{Tools: tools}.ToolsJSON()
	if err != nil { // cannot happen: Parameters maps marshal fine
		return json.RawMessage("[]")
	}
	return data
}

// readStdin drains stdin, or returns "" when stdin is nil (interactive
// terminal: nothing is being piped).
func readStdin(r io.Reader) (string, error) {
	if r == nil {
		return "", nil
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return "", fmt.Errorf("read stdin: %w", err)
	}
	return string(data), nil
}

// responseMap builds the data object the -json flag emits: the matched
// tool(s) and arguments, confidence vs. the threshold, suppressed calls,
// reasoning, the effective facts, the prompt and the full toolset that was
// loaded.
func responseMap(resp *needle.Response, names []string, above bool, facts, factsSource, prompt, toolsPath string, tools []needle.Tool, threshold float64) map[string]any {
	var matched any // null when nothing matched
	if len(names) > 0 {
		matched = names[0]
	}
	out := map[string]any{
		"type":            resp.Type,
		"success":         resp.Success,
		"matched":         len(names) > 0 && above,
		"matched_tool":    matched,
		"matched_tools":   names,
		"function_calls":  resp.FunctionCalls,
		"above_threshold": above,
		"confidence": map[string]any{
			"value":     resp.Confidence,
			"available": resp.HasConfidence,
			"threshold": threshold,
		},
		"suppressed_calls": resp.SuppressedCalls,
		"reasoning":        resp.Reasoning,
		"raw":              resp.Raw,
		"prompt":           prompt,
		"facts":            facts,
		"facts_source":     factsSource,
		"tools_path":       toolsPath,
		"tools":            toolsJSON(tools),
		"prefill_tps":      resp.PrefillTPS,
		"decode_tps":       resp.DecodeTPS,
	}
	if resp.Error != "" || resp.ErrorCode != "" {
		out["error"] = resp.Error
		out["error_code"] = resp.ErrorCode
	}
	return out
}

// emitJSON prints the responseMap as one indented JSON object.
func emitJSON(w io.Writer, resp *needle.Response, names []string, above bool, facts, factsSource, prompt, toolsPath string, tools []needle.Tool, threshold float64) error {
	out := responseMap(resp, names, above, facts, factsSource, prompt, toolsPath, tools, threshold)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

// emitFormat renders the -format template against the first function call
// and the full response object, then writes the result to stdout.
func emitFormat(w io.Writer, resp *needle.Response, names []string, above bool, facts, factsSource, prompt, toolsPath string, tools []needle.Tool, threshold float64, tmpl *template.Template) error {
	var (
		tool string
		args map[string]any
	)
	if len(resp.FunctionCalls) > 0 {
		tool = resp.FunctionCalls[0].Name
		args = resp.FunctionCalls[0].Arguments
	}
	if args == nil {
		args = map[string]any{}
	}
	data := map[string]any{
		"Tool":       tool,
		"Args":       args,
		"Confidence": resp.Confidence,
		"Success":    resp.Success,
		"Raw":        responseMap(resp, names, above, facts, factsSource, prompt, toolsPath, tools, threshold),
	}
	var buf strings.Builder
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("execute format template: %w", err)
	}
	out := buf.String()
	if out != "" && !strings.HasSuffix(out, "\n") {
		out += "\n"
	}
	if _, err := io.WriteString(w, out); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}
