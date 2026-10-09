package pure

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// layerW holds one transformer block's dequantized weights, row-major.
type layerW struct {
	normIn        []float32 // 768
	qw, kw, vw    []float32 // 576×768, 96×768, 128×768
	qTaps, kTaps  []float32 // taps×dim
	vTaps         []float32
	qNorm, kNorm  []float32 // 48
	gateW, outW   []float32 // 768×768
	postNorm      []float32 // 768
	preHada       []float32 // 768
	attnGate      float32
	d1, d2, b2    []float32
	d3, d4        []float32
	w1a, w1b, w2a []float32 // 32×32
	w2b, w3a, w3b []float32
	condV         []float32 // 768×8
	condU         []float32 // 8×1024
}

// engramW holds one engram site.
type engramW struct {
	tables  []float32 // (numTables*slots)×subDim
	keyProj []float32 // 768×768
	valProj []float32 // 768×768
	taps    []float32 // convTaps×768
}

type confHeadW struct {
	probes  []float32 // ((L+1)*k)×768
	gain    []float32 // (L+1)×k
	query   []float32 // q×768
	rowBias []float32 // q×(L+1)×k
	proj    []float32 // 1×(q*768)
	bias    float32
}

// model is the fully dequantized network (float32 throughout).
type model struct {
	geom     geometry
	codebook []float32
	tok      *tokenizer

	embedding []float32 // vocab×d
	layers    []*layerW

	aPre, aPost, aRes []float32 // L
	bPre, bPost       []float32 // L×lanes
	bRes              []float32 // L×lanes×lanes
	phiPre, phiPost   []float32 // L×lanes×(lanes*d)
	phiRes            []float32 // L×lanes×(lanes*d) rows per layer×lanes×lanes
	hadaP1, hadaP2    []float32 // hada_n index vectors
	engrams           []*engramW
	finalNorm         []float32
	conf              *confHeadW // nil when the archive carries no head

	// derived dims
	qDim, kDim, vDim, outDim int
	probeK, probeQ           int // 4, 4 for the confidence head
}

func loadModel(data []byte) (*model, error) {
	g, codebook, recs, err := parseHeader(data)
	if err != nil {
		return nil, err
	}
	m := &model{geom: g, codebook: codebook}
	m.qDim = g.NumHeads * g.QKHeadDim
	m.kDim = g.NumKVHeads * g.QKHeadDim
	m.vDim = g.NumKVHeads * g.VHeadDim
	m.outDim = g.NumHeads * g.VHeadDim
	m.probeK, m.probeQ = 4, 4

	pos := 0
	next := func(wantDtype byte, wantBits uint32) (dirRecord, error) {
		if pos >= len(recs) {
			return dirRecord{}, fmt.Errorf("pure: tensor stream exhausted at position %d", pos)
		}
		r := recs[pos]
		pos++
		if r.dtype != wantDtype || (wantBits != 0 && r.bits != wantBits) {
			return r, fmt.Errorf("pure: tensor %d dtype/bits mismatch: got (%d,%d) want (%d,%d)",
				pos-1, r.dtype, r.bits, wantDtype, wantBits)
		}
		return r, nil
	}
	anyCQ := func(r dirRecord) []float32 {
		mat, err := dequantCQ(data[r.offset:r.offset+r.nbytes], int(r.shape[0]), int(r.shape[1]), r.bits, r.group, codebook)
		if err != nil {
			panic(err)
		}
		return mat
	}
	f16v := func(r dirRecord) []float32 {
		return decodeFP16(data[r.offset : r.offset+r.nbytes])
	}

	// 0: embedding (CQ at 4 bits per the shipped file, width read from record)
	r, err := next(dtCQ, 0)
	if err != nil {
		return nil, err
	}
	m.embedding = anyCQ(r)

	// per layer: 27 tensors
	m.layers = make([]*layerW, g.NumLayers)
	for l := 0; l < g.NumLayers; l++ {
		L := &layerW{}
		must := func(dt byte, bits uint32) []float32 {
			rr, e := next(dt, bits)
			if e != nil {
				panic(e)
			}
			switch dt {
			case dtCQ:
				return anyCQ(rr)
			case dtFP16:
				return f16v(rr)
			case dtFP32:
				return decodeFP32(data[rr.offset : rr.offset+rr.nbytes])
			}
			panic("pure: unknown dtype request")
		}
		L.normIn = must(dtFP16, 0)
		L.qw = must(dtCQ, 0)
		L.kw = must(dtCQ, 0)
		L.vw = must(dtCQ, 0)
		if g.QKVConvTaps > 0 {
			L.qTaps = must(dtFP16, 0)
			L.kTaps = must(dtFP16, 0)
			L.vTaps = must(dtFP16, 0)
		}
		L.qNorm = must(dtFP16, 0)
		L.kNorm = must(dtFP16, 0)
		L.gateW = must(dtCQ, 0)
		L.outW = must(dtCQ, 0)
		L.postNorm = must(dtFP16, 0)
		agRec, e := next(dtFP16, 0)
		if e != nil {
			panic(e)
		}
		L.attnGate = f16v(agRec)[0]
		L.preHada = must(dtFP16, 0)
		L.d1 = must(dtFP16, 0)
		L.d2 = must(dtFP16, 0)
		L.b2 = must(dtFP16, 0)
		L.d3 = must(dtFP16, 0)
		L.d4 = must(dtFP16, 0)
		L.w1a = must(dtFP16, 0)
		L.w1b = must(dtFP16, 0)
		L.w2a = must(dtFP16, 0)
		L.w2b = must(dtFP16, 0)
		L.w3a = must(dtFP16, 0)
		L.w3b = must(dtFP16, 0)
		L.condV = must(dtFP16, 0)
		L.condU = must(dtFP16, 0)
		m.layers[l] = L
	}

	// mHC: 9 tensors
	{
		get := func(dt byte) []float32 {
			rr, e := next(dt, 0)
			if e != nil {
				panic(e)
			}
			if dt == dtCQ {
				return anyCQ(rr)
			}
			return f16v(rr)
		}
		m.aPre = get(dtFP16)
		m.aPost = get(dtFP16)
		m.aRes = get(dtFP16)
		m.bPre = get(dtFP16)
		m.bPost = get(dtFP16)
		m.bRes = get(dtFP16)
		m.phiPre = get(dtCQ)
		m.phiPost = get(dtCQ)
		m.phiRes = get(dtCQ)
	}
	{
		r1, e := next(dtFP32, 0)
		if e != nil {
			return nil, e
		}
		m.hadaP1 = decodeFP32(data[r1.offset : r1.offset+r1.nbytes])
		r2, e := next(dtFP32, 0)
		if e != nil {
			return nil, e
		}
		m.hadaP2 = decodeFP32(data[r2.offset : r2.offset+r2.nbytes])
	}

	// engram sites
	m.engrams = make([]*engramW, len(g.Sites))
	for s := range g.Sites {
		E := &engramW{}
		rr, _ := next(dtCQ, 0)
		E.tables = anyCQ(rr)
		rr, _ = next(dtCQ, 0)
		E.keyProj = anyCQ(rr)
		rr, _ = next(dtCQ, 0)
		E.valProj = anyCQ(rr)
		rr, _ = next(dtFP16, 0)
		E.taps = f16v(rr)
		m.engrams[s] = E
	}

	rr, _ := next(dtFP16, 0)
	m.finalNorm = f16v(rr)

	// probe heads: manifest + 6 per head; shipped file has one confidence head.
	remaining := len(recs) - pos
	if remaining > 1 {
		mn, e := next(dtFP16, 0)
		if e != nil {
			return nil, e
		}
		manifest := f16v(mn)
		nHeads := int((len(recs) - pos - 1) / 6)
		for h := 0; h < nHeads; h++ {
			code := 0
			if h < len(manifest) {
				code = int(manifest[h])
			}
			H := &confHeadW{}
			rr, _ := next(dtCQ, 0)
			H.probes = anyCQ(rr)
			rr, _ = next(dtFP16, 0)
			H.gain = f16v(rr)
			rr, _ = next(dtCQ, 0)
			H.query = anyCQ(rr)
			rr, _ = next(dtFP16, 0)
			H.rowBias = f16v(rr)
			rr, _ = next(dtCQ, 0)
			H.proj = anyCQ(rr)
			rr, _ = next(dtFP16, 0)
			H.bias = f16v(rr)[0]
			if code == 2 { // confidence
				m.conf = H
			}
		}
	}

	// last record: RAW tokenizer
	rt, e := next(dtRAW, 0)
	if e != nil {
		return nil, e
	}
	m.tok, err = newTokenizer(data[rt.offset : rt.offset+rt.nbytes])
	if err != nil {
		return nil, err
	}
	if pos != len(recs) {
		return nil, fmt.Errorf("pure: %d directory records left unread", len(recs)-pos)
	}
	return m, nil
}

// gemv computes y = W·x for a row-major rows×cols matrix. Rows are split
// across workers when the matvec is large enough to pay for the fan-out.
func gemv(w, x, y []float32, rows, cols int) {
	work := rows * cols
	if work < 1<<14 {
		gemvRange(w, x, y, 0, rows, cols)
		return
	}
	nw := runtime.NumCPU()
	if nw > rows {
		nw = rows
	}
	if nw > 16 {
		nw = 16
	}
	chunk := (rows + nw - 1) / nw
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		lo := k * chunk
		if lo >= rows {
			break
		}
		hi := lo + chunk
		if hi > rows {
			hi = rows
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvRange(w, x, y, lo, hi, cols)
		}(lo, hi)
	}
	wg.Wait()
}

func gemvRange(w, x, y []float32, lo, hi, cols int) {
	for r := lo; r < hi; r++ {
		row := w[r*cols : (r+1)*cols]
		var acc float32
		for c, v := range row {
			acc += v * x[c]
		}
		y[r] = acc
	}
}

func dot(a, b []float32) float32 {
	var acc float32
	for i, v := range a {
		acc += v * b[i]
	}
	return acc
}

func rmsUnit(x []float32, out []float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	s := float32(1.0) / float32(math.Sqrt(float64(sum)/float64(len(x))+1e-6))
	for i, v := range x {
		out[i] = v * s
	}
}

func zcrmsNorm(x, scale []float32, out []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(sum/float64(len(x)) + 1e-6))
	for i, v := range x {
		out[i] = (1 + scale[i]) * v / rms
	}
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func softmaxInPlace(v []float32) {
	maxv := v[0]
	for _, x := range v[1:] {
		if x > maxv {
			maxv = x
		}
	}
	var sum float64
	for i, x := range v {
		e := float32(math.Exp(float64(x - maxv)))
		v[i] = e
		sum += float64(e)
	}
	inv := float32(1.0 / sum)
	for i := range v {
		v[i] *= inv
	}
}
