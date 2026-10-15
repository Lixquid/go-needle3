//go:build cgo && linux

package needle_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"

	needle "github.com/lixquid/go-needle3"
	"github.com/lixquid/go-needle3/engine/cgo"
	"github.com/lixquid/go-needle3/engine/pure"
	nmodel "github.com/lixquid/go-needle3/model"
)

// TestFuzzDifferential runs a seeded pseudo-random prompt corpus through both
// the C engine (engine/cgo) and the pure-Go port, comparing type, parsed tool
// calls, and generated reasoning text per prompt. The C ABI exposes no knob
// to disable the engine's internal activation/KV quantization, so this
// measures end-to-end agreement under the engine's native quantized mode;
// divergences are expected where the engine deviates from the JAX reference,
// not the port.
func TestFuzzDifferential(t *testing.T) {
	tools := demoTools()
	ca, err := cgo.New(needle.Options{Weights: nmodel.Needle3, Tools: tools, MaxNewTokens: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer ca.Close()
	pa, err := pure.New(needle.Options{Weights: nmodel.Needle3, Tools: tools, MaxNewTokens: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer pa.Close()

	cases := fuzzCorpus()
	results := make([][2]*needle.Response, len(cases))
	var (
		bothErr, oneErr            int
		typeEq, callsEq, exact     int
		confDeltaSum, confDeltaMax float64
		divergences                []string
	)
	for idx, c := range cases {
		pa.Reset()
		ca.Reset()
		pr, perr := pa.Complete(c.prompt)
		cr, cerr := ca.Complete(c.prompt)
		switch {
		case perr != nil && cerr != nil:
			bothErr++
			continue
		case perr != nil || cerr != nil:
			oneErr++
			divergences = append(divergences, fmt.Sprintf("[%s] only one backend errored: pure=%v cgo=%v", c.name, perr, cerr))
			continue
		}
		if pr.Type == cr.Type {
			typeEq++
		} else {
			divergences = append(divergences, fmt.Sprintf("[%s] type %q vs %q (pure vs cgo)", c.name, pr.Type, cr.Type))
		}
		if canonCalls(pr) == canonCalls(cr) {
			callsEq++
		}
		reasonEq := pr.Reasoning == cr.Reasoning
		if reasonEq && pr.Type == cr.Type && canonCalls(pr) == canonCalls(cr) {
			exact++
		}
		d := pr.Confidence - cr.Confidence
		if d < 0 {
			d = -d
		}
		confDeltaSum += d
		if d > confDeltaMax {
			confDeltaMax = d
		}
		if !reasonEq {
			pfx := commonPrefixLen(pr.Reasoning, cr.Reasoning)
			divergences = append(divergences, fmt.Sprintf("[%s] reasoning diverges at char %d/%d (%d%%): pure=%q cgo=%q",
				c.name, pfx, len(cr.Reasoning), pct(pfx, len(cr.Reasoning)), trim(pr.Reasoning, 90), trim(cr.Reasoning, 90)))
		} else if canonCalls(pr) != canonCalls(cr) || pr.Type != cr.Type {
			divergences = append(divergences, fmt.Sprintf("[%s] same reasoning, different outcome: pure type=%s calls=%s conf=%.3f | cgo type=%s calls=%s conf=%.3f",
				c.name, pr.Type, canonCalls(pr), pr.Confidence, cr.Type, canonCalls(cr), cr.Confidence))
		}
		results[idx] = [2]*needle.Response{pr, cr}
		t.Logf("%-16s type=%v calls=%v reasoning=%v conf %.3f/%.3f",
			c.name, pr.Type == cr.Type, canonCalls(pr) == canonCalls(cr), reasonEq, pr.Confidence, cr.Confidence)
	}

	t.Logf("=== fuzz differential: %d cases ===", len(cases))
	t.Logf("type agreement    %d/%d", typeEq, len(cases))
	t.Logf("calls agreement   %d/%d", callsEq, len(cases))
	t.Logf("fully exact       %d/%d", exact, len(cases))
	t.Logf("error divergences one-sided=%d both=%d", oneErr, bothErr)
	t.Logf("confidence delta  mean=%.4f max=%.4f", confDeltaSum/float64(len(cases)), confDeltaMax)
	for _, d := range divergences {
		t.Log("DIVERGE " + d)
	}
	if err := writeFuzzReport("/tmp/fuzzdiff_report.txt", cases, results); err != nil {
		t.Error(err)
	}

	// Determinism guard: the pure engine must be byte-reproducible.
	for _, p := range []string{"how hot is it in Oslo right now", "set the bedroom brightness to 40 percent", "☀️ tomorrow?"} {
		pa.Reset()
		r1, err := pa.Complete(p)
		if err != nil {
			t.Fatal(err)
		}
		pa.Reset()
		r2, err := pa.Complete(p)
		if err != nil {
			t.Fatal(err)
		}
		if r1.Raw != r2.Raw {
			t.Errorf("pure nondeterministic on %q: %q vs %q", p, trim(r1.Raw, 60), trim(r2.Raw, 60))
		}
	}
}

func demoTools() []needle.Tool {
	return []needle.Tool{
		{
			Name:        "get_weather",
			Description: "Current weather for a city.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string", "description": "City name"},
				},
				"required": []string{"city"},
			},
		},
		{
			Name:        "control_light",
			Description: "Turn a room light on or off, optionally setting brightness.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"room":       map[string]any{"type": "string"},
					"on":         map[string]any{"type": "boolean"},
					"brightness": map[string]any{"type": "number"},
				},
				"required": []string{"room"},
			},
		},
	}
}

func fuzzCorpus() []struct{ name, prompt string } {
	rnd := rand.New(rand.NewSource(42))
	add := func(name, prompt string) []struct{ name, prompt string } {
		return []struct{ name, prompt string }{{name, prompt}}
	}
	var cases []struct{ name, prompt string }
	weather := []string{"what is the weather in %s?", "how hot is it in %s right now", "is it raining in %s", "weather forecast for %s please", "check the weather in %s", "what's it like in %s right now?"}
	cities := []string{"Paris", "Lagos", "Oslo", "Tokyo", "Mumbai", "Reykjavik"}
	for i, city := range cities {
		cases = append(cases, add("w-"+city, fmt.Sprintf(weather[i], city))...)
	}
	lights := []string{"turn on the kitchen lights", "set the bedroom brightness to 40 percent", "dim the living room to 30", "turn off the hallway light", "brighten the office to 80 percent"}
	for i, s := range lights {
		cases = append(cases, add("l-"+fmt.Sprint(i), s)...)
	}
	vocab := []string{"purple", "mountain", "rainforest", "humid", "temperature", "light", "weather", "kitchen", "quiet", "loud", "seven", "garden"}
	for i := 0; i < 8; i++ {
		n := 1 + rnd.Intn(4)
		var w []string
		for j := 0; j < n; j++ {
			w = append(w, vocab[rnd.Intn(len(vocab))])
		}
		s := strings.Join(w, " ")
		if i%2 == 0 {
			s += "?"
		} else {
			s += "."
		}
		cases = append(cases, add("rnd-"+strconvItoa(i), s)...)
	}
	for i, s := range []string{"hi", "ok.", "42", "???", "New York City", "the"} {
		cases = append(cases, add("edge-"+strconvItoa(i), s)...)
	}
	for i, s := range []string{"温度は？", "café naïve résumé", "set temp to 23.5°C", "☀️ tomorrow?", "Är det varmt i Paris?"} {
		cases = append(cases, add("uni-"+strconvItoa(i), s)...)
	}
	longBase := strings.Repeat("The weather in Paris is nice and calm. ", 18)
	for i, q := range []string{"turn the lights on now.", "how hot is it in Oslo?", "close the window.", "quiet please.", "seven gardens."} {
		cases = append(cases, add("long-"+strconvItoa(i), longBase+" "+q)...)
	}
	return cases
}

func strconvItoa(i int) string { return fmt.Sprint(i) }

func canonCalls(r *needle.Response) string {
	var parts []string
	for _, c := range r.FunctionCalls {
		b, _ := json.Marshal(c.Arguments)
		parts = append(parts, c.Name+"("+string(b)+")")
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

func commonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func pct(a, b int) int {
	if b == 0 {
		return 100
	}
	return 100 * a / b
}

func writeFuzzReport(path string, cases []struct{ name, prompt string }, res [][2]*needle.Response) error {
	var b strings.Builder
	for i, c := range cases {
		fmt.Fprintf(&b, "== %s\nprompt: %q\n", c.name, c.prompt)
		for _, side := range []string{"pure", "cgo"} {
			var r *needle.Response
			if side == "pure" {
				r = res[i][0]
			} else {
				r = res[i][1]
			}
			if r == nil {
				fmt.Fprintf(&b, "%s: <error>\n", side)
				continue
			}
			fmt.Fprintf(&b, "%s: type=%s calls=%s conf=%.3f reasoning=%q\n", side, r.Type, canonCalls(r), r.Confidence, trim(r.Reasoning, 140))
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func trim(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
