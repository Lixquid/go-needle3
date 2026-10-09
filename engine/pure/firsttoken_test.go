package pure

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

func TestFirstTokenTop5(t *testing.T) {
	m := loadReal(t)
	toolsJSON := `[{"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"properties": {"city": {"description": "city name", "type": "string"}}, "required": ["city"], "type": "object"}}, {"name": "set_lights", "description": "Turn a room's lights on/off and optionally set brightness.", "parameters": {"properties": {"brightness": {"maximum": 100, "minimum": 0, "type": "integer"}, "on": {"description": "turn on or off", "type": "boolean"}, "room": {"description": "room name", "type": "string"}}, "required": ["room", "on"], "type": "object"}}]`
	prefixText := "<|im_start|>system\ndate: 2026-10-08 Thu 09:41; device: laptop<|im_end|>\n<|im_start|>user\n<tools>" + toolsJSON + "</tools>"
	prefix := append([]int{TokBOS}, m.tok.encode(prefixText)...)
	t.Logf("prefix len=%d ids=%v", len(prefix), prefix)
	turnText := "\nturn on the kitchen lights<|im_end|>\n<|im_start|>assistant\n"
	turn := m.tok.encode(turnText)
	t.Logf("turn ids=%v", turn)

	{
		joined := append(append(append([]int{}, prefix...), turn...), TokThinkStart)
		b, _ := json.Marshal(joined)
		os.WriteFile("/tmp/oracle_ids.json", b, 0o644)
	}
	st := newRunState(m, len(prefix)+len(turn)+16)
	var logits []float32
	for _, id := range prefix {
		logits = st.step(id)
	}
	t.Logf("logits after prefix: finite? %v", logitsFinite(logits))
	for _, id := range turn {
		logits = st.step(id)
	}
	t.Logf("logits after turn: finite? %v", logitsFinite(logits))
	logits = st.step(TokThinkStart)
	dumpCells(st, logits)
	top := topK(logits, 5)
	for _, p := range top {
		t.Logf("top: id=%d p=%.4f piece=%q", p.id, p.p, m.tok.pieces[p.id])
	}
}

func logitsFinite(v []float32) bool {
	for _, x := range v {
		if x != x {
			return false
		}
	}
	return true
}

type idp struct {
	id int
	p  float64
}

func topK(logits []float32, k int) []idp {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	out := make([]idp, 0, k)
	var sum float64
	for _, i := range idx {
		sum += expF(float64(logits[i] - logits[idx[0]]))
	}
	for i := 0; i < k; i++ {
		out = append(out, idp{idx[i], expF(float64(logits[idx[i]]-logits[idx[0]])) / sum})
	}
	return out
}

func expF(x float64) float64 {
	if x < -60 {
		return 0
	}
	// cheap exp via fmt is fine for tests
	var f float64
	fmt.Sscanf(fmt.Sprintf("%g", expGo(x)), "%g", &f)
	return f
}

func expGo(x float64) float64 {
	// simple exp
	res := 1.0
	term := 1.0
	for i := 1; i < 40; i++ {
		term *= x / float64(i)
		res += term
	}
	return res
}

func dumpCells(st *runState, logits []float32) {
	m := st.m
	d := m.geom.DModel
	T := st.cellEnd()
	flat := make([]float32, T*21*d)
	copy(flat, st.cells[:T*21*d])
	if err := os.WriteFile("/tmp/go_cells.bin", flatBytes(flat), 0o644); err != nil {
		panic(err)
	}
	os.WriteFile("/tmp/go_logits.bin", flatBytes(logits), 0o644)
	fmt.Printf("go cells final emb[:4]: %v\n", st.cells[(T-1)*21*d:(T-1)*21*d+4])
	for _, l := range []int{1, 2, 5, 10, 19, 20} {
		o := (T-1)*21*d + l*d
		fmt.Printf("go cells final layer%d mean[:4]: %v\n", l, st.cells[o:o+4])
	}
}

func flatBytes(f []float32) []byte {
	out := make([]byte, 4*len(f))
	for i, v := range f {
		b := math.Float32bits(v)
		out[4*i] = byte(b)
		out[4*i+1] = byte(b >> 8)
		out[4*i+2] = byte(b >> 16)
		out[4*i+3] = byte(b >> 24)
	}
	return out
}
