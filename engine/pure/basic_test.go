package pure

import (
	"os"
	"testing"
)

func loadReal(t *testing.T) *model {
	t.Helper()
	data, err := os.ReadFile("../../model/needle3.cact")
	if err != nil {
		t.Skip("model file not found")
	}
	m, err := loadModel(data)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return m
}

func TestShapes(t *testing.T) {
	m := loadReal(t)
	L := m.layers[0]
	t.Logf("emb=%d qw=%d kw=%d vw=%d gate=%d out=%d",
		len(m.embedding), len(L.qw), len(L.kw), len(L.vw), len(L.gateW), len(L.outW))
	t.Logf("taps q=%d k=%d v=%d", len(L.qTaps), len(L.kTaps), len(L.vTaps))
	t.Logf("condV=%d condU=%d d1=%d w1a=%d p1=%d", len(L.condV), len(L.condU), len(L.d1), len(L.w1a), len(m.hadaP1))
	t.Logf("phiPre=%d phiPost=%d phiRes=%d bRes=%d", len(m.phiPre), len(m.phiPost), len(m.phiRes), len(m.bRes))
	E := m.engrams[0]
	t.Logf("tables=%d keyProj=%d valProj=%d taps=%d", len(E.tables), len(E.keyProj), len(E.valProj), len(E.taps))
	t.Logf("conf probes=%d gain=%d query=%d rowBias=%d proj=%d", len(m.conf.probes), len(m.conf.gain), len(m.conf.query), len(m.conf.rowBias), len(m.conf.proj))
	t.Logf("normIn=%d qNorm=%d post=%d preHada=%d final=%d", len(L.normIn), len(L.qNorm), len(L.postNorm), len(L.preHada), len(m.finalNorm))
}

func TestTokenizer(t *testing.T) {
	m := loadReal(t)
	ids := m.tok.encode("turn on the kitchen lights")
	t.Logf("ids=%v", ids)
	if len(ids) == 0 {
		t.Fatal("empty encode")
	}
	back := m.tok.decode(ids)
	t.Logf("roundtrip: %q", back)
	if back != "turn on the kitchen lights" {
		t.Errorf("roundtrip mismatch: %q", back)
	}
	callIDs := m.tok.encode("<tool_call>")
	if len(callIDs) != 1 || callIDs[0] != TokCallStart {
		t.Errorf("marker encode: %v", callIDs)
	}
}

func TestDequantGolden(t *testing.T) {
	m := loadReal(t)
	L := m.layers[0]
	E := m.engrams[0]
	check := func(name string, got []float32, want []float64) {
		for i, w := range want {
			d := float64(got[i]) - w
			if d < 0 {
				d = -d
			}
			if d > 1e-3 {
				t.Errorf("%s[%d]: got %v want %v", name, i, got[i], w)
			}
		}
	}
	check("emb", m.embedding[2*768:2*768+6],
		[]float64{-4.81263, 8.40596, -8.61934, 36.4926, 24.71906, -7.32272})
	check("q_proj", L.qw[:5], []float64{-0.05827, 0.32691, 0.04006, 0.26974, -0.47581})
	check("tables", E.tables[:5], []float64{-3.35547, -6.19906, 2.66802, -0.50029, 0.94934})
	check("phiPre", m.phiPre[:5], []float64{-1.71014, -0.85634, -0.55797, 0.28407, 0.01142})
}
