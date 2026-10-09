package pure

import (
	"testing"

	"github.com/Zetaphor/needle-go"
	nmodel "github.com/Zetaphor/needle-go/model"
)

func demoTools() []needle.Tool {
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
		},
	}
}

func TestGenerateDump(t *testing.T) {
	a, err := New(needle.Options{Tools: demoTools(), System: "date: 2026-10-08 Thu 09:41; device: laptop", Weights: nmodel.Needle3})
	if err != nil {
		t.Fatal(err)
	}
	r, err := a.Complete("turn on the kitchen lights")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("type=%s err=%v reasoning=%q\nraw=%q\ncalls=%v conf=%v", r.Type, r.Error, r.Reasoning, r.Raw, r.FunctionCalls, r.Confidence)
}

func TestGenerateDump2(t *testing.T) {
	a, _ := New(needle.Options{Tools: demoTools(), System: "date: 2026-10-08 Thu 09:41; device: laptop", Weights: nmodel.Needle3})
	for _, q := range []string{
		"what's it like in Lagos right now?",
		"the Purple Mountain Rainforest is very humid",
	} {
		r, err := a.Complete(q)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Q=%q\nraw=%q\ncalls=%v conf=%v", q, r.Raw, r.FunctionCalls, r.Confidence)
	}
}
