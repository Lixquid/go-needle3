package pure

import (
	"testing"

	needle "github.com/Zetaphor/needle-go"
	nmodel "github.com/Zetaphor/needle-go/model"
)

func TestEncIDs(t *testing.T) {
	a, err := New(needle.Options{Weights: nmodel.Needle3})
	if err != nil {
		t.Fatal(err)
	}
	t.Log(a.m.tok.encode("dim the living room to 30"))
}
