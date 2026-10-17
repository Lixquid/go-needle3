// Command needle-pure demonstrates the cgo-backed Needle 3 agent with the
// model embedded in the binary.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/lixquid/go-needle3"
	"github.com/lixquid/go-needle3/engine/pure"
	"github.com/lixquid/go-needle3/model"
)

func main() {
	verbose := flag.Bool("v", false, "print raw engine JSON for each turn")
	flag.Parse()

	agent, err := pure.New(needle.Options{
		Tools:   tools(),
		System:  "date: 2026-10-08 Thu 09:41; device: laptop",
		Weights: model.Needle3,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer agent.Close()
	fmt.Printf("model loaded: %d kinds active\n\n", 1)

	demos := []struct {
		title string
		query string
		useFn bool // run through the agentic loop (handlers execute)
	}{
		{"tool call", "turn on the kitchen lights", false},
		{"tool call", "what's it like in Lagos right now?", false},
		{"agentic loop", "set the kitchen lights on and check the weather in Oslo", true},
		{"off-topic", "the Purple Mountain Rainforest is very humid", false},
	}

	for _, d := range demos {
		fmt.Printf("== %s: %q\n", d.title, d.query)
		var (
			r   *needle.Response
			err error
		)
		if d.useFn {
			r, err = agent.Run(d.query)
		} else {
			r, err = agent.Complete(d.query)
		}
		if err != nil {
			fmt.Println("  error:", err)
			continue
		}
		printResponse(r, *verbose)
		if err := agent.Reset(); err != nil {
			fmt.Println("  reset:", err)
		}
		fmt.Println()
	}

	vec, err := agent.Embed("dim the living room to 30")
	if err != nil {
		fmt.Println("embed error:", err)
		return
	}
	fmt.Printf("== embed: 3072-dim? %v, first values: %.4f %.4f %.4f\n", len(vec) == 3072, vec[0], vec[1], vec[2])

	// Extraction binds its own toolset, so it wants the engine slot to itself.
	agent.Close()
	extractionDemo(*verbose)
}

func printResponse(r *needle.Response, verbose bool) {
	fmt.Printf("  type=%s confidence=%.3f\n", r.Type, r.Confidence)
	if len(r.FunctionCalls) == 0 {
		if len(r.SuppressedCalls) > 0 {
			fmt.Printf("  no calls (below confidence gate; suppressed %d)\n", len(r.SuppressedCalls))
		} else {
			fmt.Println("  no calls (off-topic or respond)")
		}
	}
	for _, c := range r.FunctionCalls {
		args, _ := json.Marshal(c.Arguments)
		fmt.Printf("  call: %s %s\n", c.Name, args)
	}
	for _, res := range r.Results {
		out, _ := json.Marshal(res.Output)
		fmt.Printf("  executed: %s -> %s\n", res.Call.Name, out)
	}
	if r.Reasoning != "" {
		fmt.Printf("  reasoning: %s\n", r.Reasoning)
	}
	if verbose {
		b, _ := json.Marshal(r)
		fmt.Printf("  raw: %s\n", b)
	}
}

func tools() []needle.Tool {
	return []needle.Tool{
		{
			Name:        "get_weather",
			Description: "Get the current weather for a city.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string", "description": "city name"},
				},
				"required": []string{"city"},
			},
			Fn: func(args map[string]any) (any, error) {
				city, _ := args["city"].(string)
				return map[string]any{"city": city, "temp_c": 27, "sky": "clear"}, nil
			},
		},
		{
			Name:        "set_lights",
			Description: "Turn a room's lights on/off and optionally set brightness.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"room":       map[string]any{"type": "string", "description": "room name"},
					"on":         map[string]any{"type": "boolean", "description": "turn on or off"},
					"brightness": map[string]any{"type": "integer", "minimum": 0, "maximum": 100},
				},
				"required": []string{"room", "on"},
			},
			Fn: func(args map[string]any) (any, error) {
				return map[string]any{"ok": true, "room": args["room"], "on": args["on"]}, nil
			},
		},
	}
}

// invoice is the extraction demo record: a money amount and a date that the
// strict grounding check can hold to the input.
type invoice struct {
	Vendor  string      `json:"vendor" description:"who issued the invoice"`
	Total   float64     `json:"total,omitempty" description:"amount due"`
	DueDate needle.Date `json:"due_date,omitempty" description:"when it is due"`
}

// extractionDemo is the model's other job: a schema goes in with the text and
// typed fields come back. The one-liner form is pure.ExtractAs[invoice](text,
// opts); this uses the generic form so the values still print when the strict
// check finds one of them ungrounded.
func extractionDemo(verbose bool) {
	const text = "Invoice from Acme for $1,200, due on 5th September 2031"
	fmt.Printf("\n== extraction: %q\n", text)

	opts := needle.Options{Weights: model.Needle3, System: "date: 2026-10-08 Thu 09:41; device: laptop"}
	ex, err := pure.Extract(text, needle.SchemaOf[invoice](), opts)
	if err != nil {
		var exErr *needle.ExtractionError
		if !errors.As(err, &exErr) {
			fmt.Println("  error:", err)
			return
		}
		fmt.Println(" ", exErr)
	}
	var inv invoice
	if err := ex.Decode(&inv); err != nil {
		fmt.Println("  decode:", err)
		return
	}
	fmt.Printf("  vendor=%q total=%.2f due=%s confidence=%.3f\n", inv.Vendor, inv.Total, inv.DueDate, ex.Confidence)
	if verbose {
		b, _ := json.Marshal(ex.Response)
		fmt.Printf("  raw: %s\n", b)
	}
}
