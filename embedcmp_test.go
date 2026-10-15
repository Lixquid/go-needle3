//go:build cgo && linux

package needle_test

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	needle "github.com/lixquid/go-needle3"
	"github.com/lixquid/go-needle3/engine/cgo"
	"github.com/lixquid/go-needle3/engine/pure"
	nmodel "github.com/lixquid/go-needle3/model"
)

func TestEmbedCosine(t *testing.T) {
	text := "The Purple Mountain Rainforest is very humid"
	ca, err := cgo.New(needle.Options{Weights: nmodel.Needle3})
	if err != nil {
		t.Fatal(err)
	}
	defer ca.Close()
	pa, err := pure.New(needle.Options{Weights: nmodel.Needle3})
	if err != nil {
		t.Fatal(err)
	}
	defer pa.Close()
	ev, err := ca.Embed(text)
	if err != nil {
		t.Fatal(err)
	}
	pv, err := pa.Embed(text)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("engine dim=%d pure dim=%d", len(ev), len(pv))
	var dot, na, nb float64
	for i := range pv {
		dot += float64(pv[i]) * float64(ev[i])
		na += float64(pv[i]) * float64(pv[i])
		nb += float64(ev[i]) * float64(ev[i])
	}
	t.Logf("cosine = %.4f", dot/(math.Sqrt(na)*math.Sqrt(nb)+1e-12))
	writeF32(t, "/tmp/eng_embed.bin", ev)
	writeF32(t, "/tmp/pure_embed.bin", pv)
}

func writeF32(t *testing.T, p string, v []float32) {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
