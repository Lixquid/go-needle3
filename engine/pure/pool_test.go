package pure

import (
	"math"
	"testing"

	needle "github.com/lixquid/go-needle3"
	nmodel "github.com/lixquid/go-needle3/model"
)

// TestEmbedPool guards the probe-pool path (regression: unbounded slices in
// rmsUnit/gain application).
func TestEmbedPool(t *testing.T) {
	a := newSharedAgent(t, needle.Options{Weights: nmodel.Needle3})
	st := newRunState(a.m, 64)
	for _, id := range append([]int{TokBOS}, a.m.tok.encode("dim the living room to 30")...) {
		st.step(id)
	}
	pool := st.probePooled()
	if len(pool) != 4*a.m.geom.DModel {
		t.Fatalf("pool len = %d, want %d", len(pool), 4*a.m.geom.DModel)
	}
	v, err := a.Embed("dim the living room to 30")
	if err != nil {
		t.Fatal(err)
	}
	var norm float64
	for _, f := range v {
		norm += float64(f) * float64(f)
	}
	if math.Abs(norm-1) > 1e-3 {
		t.Errorf("embedding norm = %v, want 1", norm)
	}
}
