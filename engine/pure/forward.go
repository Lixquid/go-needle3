package pure

import (
	"math"
	"runtime"
)

// runState carries everything needed to continue generation from the last
// processed position.
type runState struct {
	m *model

	t      int   // last processed position (-1 when empty)
	tokens []int // token ids by position

	stream  []float32
	laneBuf []float32 // lanes×d residual at the current position

	kCache [][]float32 // per layer, positions × kDim
	vCache [][]float32 // per layer, positions × vDim

	qHist [][]float32 // per layer ring: taps×qDim raw projections
	kHist [][]float32
	vHist [][]float32

	egRawV [][]float32 // per engram site ring: 10×d raw value projections
	egK    [][]float32 // per site: key vector at current position
	egV    [][]float32 // per site: tapped value vector at current position

	cells []float32 // T×(L+1)×d hidden means for the probe head
	cap   int
	cellT int

	ropeCos [][]float32
	ropeSin [][]float32

	// scratch buffers (lane count is 4 in the shipped geometry)
	nx    []float32
	u     []float32
	y     []float32
	tmp   []float32
	hpre  []float32
	hpost []float32
	res   []float32
}

func newRunState(m *model, capHint int) *runState {
	g := m.geom
	if capHint < 64 {
		capHint = 64
	}
	if capHint > g.MaxSeqLen {
		capHint = g.MaxSeqLen
	}
	st := &runState{m: m, t: -1, cap: capHint}
	st.stream = make([]float32, g.MHCLanes*g.DModel)
	st.laneBuf = make([]float32, g.MHCLanes*g.DModel)
	st.kCache = make([][]float32, g.NumLayers)
	st.vCache = make([][]float32, g.NumLayers)
	st.qHist = make([][]float32, g.NumLayers)
	st.kHist = make([][]float32, g.NumLayers)
	st.vHist = make([][]float32, g.NumLayers)
	taps := g.QKVConvTaps
	if taps < 1 {
		taps = 1
	}
	for l := range st.kCache {
		st.kCache[l] = make([]float32, capHint*m.kDim)
		st.vCache[l] = make([]float32, capHint*m.vDim)
		st.qHist[l] = make([]float32, taps*m.qDim)
		st.kHist[l] = make([]float32, taps*m.kDim)
		st.vHist[l] = make([]float32, taps*m.vDim)
	}
	st.egRawV = make([][]float32, len(g.Sites))
	st.egK = make([][]float32, len(g.Sites))
	st.egV = make([][]float32, len(g.Sites))
	for s := range st.egRawV {
		st.egRawV[s] = make([]float32, 10*g.DModel)
		st.egK[s] = make([]float32, g.DModel)
		st.egV[s] = make([]float32, g.DModel)
	}
	st.cellT = g.NumLayers + 1
	st.cells = make([]float32, capHint*st.cellT*g.DModel)
	n := g.MHCLanes
	st.nx = make([]float32, n*g.DModel)
	st.u = make([]float32, g.DModel)
	st.y = make([]float32, g.DModel)
	st.tmp = make([]float32, g.DModel)
	st.hpre = make([]float32, n)
	st.hpost = make([]float32, n)
	st.res = make([]float32, n*n)
	return st
}

func (st *runState) cellEnd() int { return st.t + 1 }

func (st *runState) tokenAt(t int) int {
	if t < 0 || t >= len(st.tokens) {
		return 0
	}
	return st.tokens[t]
}

func (st *runState) ensureRope(t int) {
	for len(st.ropeCos) <= t {
		p := len(st.ropeCos)
		half := st.m.geom.QKHeadDim / 2
		c := make([]float32, half)
		s := make([]float32, half)
		for i := 0; i < half; i++ {
			freq := float32(1.0 / math.Pow(float64(st.m.geom.RopeTheta), float64(2*i)/float64(st.m.geom.QKHeadDim)))
			ang := float64(p) * float64(freq)
			c[i] = float32(math.Cos(ang))
			s[i] = float32(math.Sin(ang))
		}
		st.ropeCos = append(st.ropeCos, c)
		st.ropeSin = append(st.ropeSin, s)
	}
}

func (st *runState) ensurePos(t int) {
	if t < st.cap {
		return
	}
	nc := st.cap * 2
	if nc <= t {
		nc = t + 1
	}
	m := st.m
	for l := 0; l < m.geom.NumLayers; l++ {
		nk := make([]float32, nc*m.kDim)
		copy(nk, st.kCache[l])
		st.kCache[l] = nk
		nv := make([]float32, nc*m.vDim)
		copy(nv, st.vCache[l])
		st.vCache[l] = nv
	}
	ncells := make([]float32, nc*st.cellT*m.geom.DModel)
	copy(ncells, st.cells)
	st.cells = ncells
	st.cap = nc
}

// step feeds one token and returns next-token logits.
func (st *runState) step(token int) []float32 {
	m := st.m
	g := m.geom
	t := st.t + 1
	st.t = t
	if len(st.tokens) <= t {
		st.tokens = append(st.tokens, token)
	} else {
		st.tokens[t] = token
	}
	st.ensurePos(t)
	st.ensureRope(t)
	d := g.DModel

	// embedding × sqrt(d), broadcast to lanes
	embRow := m.embedding[token*d : (token+1)*d]
	scale := float32(math.Sqrt(float64(d)))
	for c := 0; c < g.MHCLanes; c++ {
		off := c * d
		for i, v := range embRow {
			st.stream[off+i] = v * scale
		}
	}
	// cell 0 = input embedding (scaled)
	cell0 := st.cells[t*st.cellT*d : (t+1)*st.cellT*d]
	for i, v := range embRow {
		cell0[i] = v * scale
	}

	// engram keys/values are a pure function of the token history
	st.computeEngrams(t)

	for l := 0; l < g.NumLayers; l++ {
		st.advance(l)
		base := t*st.cellT*d + (l+1)*d
		for i := 0; i < d; i++ {
			var acc float32
			for c := 0; c < g.MHCLanes; c++ {
				acc += st.stream[c*d+i]
			}
			st.cells[base+i] = acc / float32(g.MHCLanes)
		}
	}

	// final: lane mean → final norm → tied embedding head
	xn := st.tmp[:d]
	for i := 0; i < d; i++ {
		var acc float32
		for c := 0; c < g.MHCLanes; c++ {
			acc += st.stream[c*d+i]
		}
		xn[i] = acc / float32(g.MHCLanes)
	}
	zcrmsNorm(xn, m.finalNorm, xn)

	logits := make([]float32, g.OutVocab)
	gemv(m.embedding, xn, logits, g.OutVocab, d)
	return logits
}

var engramIdxScratch []int

// computeEngrams computes each site's k and tapped v for position t.
func (st *runState) computeEngrams(t int) {
	m := st.m
	g := m.geom
	if len(g.Sites) == 0 {
		return
	}
	d := g.DModel
	heads := m.engramHeads()
	stride := g.EngramSeedHds
	if stride == 0 {
		stride = heads
	}
	ntab := g.NumEngramTabs
	// n-gram slot indices are shared across sites
	if cap(engramIdxScratch) < ntab {
		engramIdxScratch = make([]int, ntab)
	}
	idxs := engramIdxScratch[:ntab]
	for tab := 0; tab < ntab; tab++ {
		oi := tab / heads
		order := g.Orders[oi]
		h := tab % heads
		if t < order-1 {
			idxs[tab] = -1
			continue
		}
		acc := uint32(0x9E3779B9) * uint32(stride*oi+h+1)
		for j := 0; j < order; j++ {
			tok := uint32(0)
			if t-j >= 0 {
				tok = uint32(st.tokenAt(t - j))
			}
			acc = (acc ^ tok) * uint32(0x01000193)
		}
		acc ^= acc >> 15
		idxs[tab] = int(acc % uint32(g.EngramSlots))
	}
	sub := g.EngramSubDim
	for s, E := range m.engrams {
		e := make([]float32, d)
		for tab := 0; tab < ntab; tab++ {
			if idxs[tab] < 0 {
				continue
			}
			rowOff := (tab*g.EngramSlots + idxs[tab]) * sub
			copy(e[tab*sub:(tab+1)*sub], E.tables[rowOff:rowOff+sub])
		}
		gemvPar2(E.keyProj, e, st.egK[s], d, d)
		vraw := make([]float32, d)
		gemvPar2(E.valProj, e, vraw, d, d)
		copy(st.egRawV[s][(t%10)*d:], vraw)
		v := st.egV[s]
		for i := range v {
			v[i] = 0
		}
		for j := 0; j < g.EngramConvTaps; j++ {
			back := j * g.EngramDilation
			if t-back < 0 {
				continue
			}
			src := st.egRawV[s][((t-back)%10)*d:]
			tw := E.taps[j*d:]
			for i := range v {
				v[i] += tw[i] * src[i]
			}
		}
	}
}

// advance applies one layer with multi-lane hyper-connections.
func (st *runState) advance(l int) {
	m := st.m
	g := m.geom
	n, d := g.MHCLanes, g.DModel
	nc := n * d

	rmsUnit(st.stream, st.nx)

	for c := 0; c < n; c++ {
		v := m.aPre[l]*dot(st.nx, m.phiPre[(l*n+c)*nc:]) + m.bPre[l*n+c] + preOff(l, c)
		st.hpre[c] = sigmoid(v)
		v = m.aPost[l]*dot(st.nx, m.phiPost[(l*n+c)*nc:]) + m.bPost[l*n+c] + postOff(l, c)
		st.hpost[c] = 2 * sigmoid(v)
	}
	for i := 0; i < d; i++ {
		var acc float32
		for c := 0; c < n; c++ {
			acc += st.hpre[c] * st.stream[c*d+i]
		}
		st.u[i] = acc
	}

	st.block(l)
	for i := 0; i < d; i++ {
		st.y[i] -= st.u[i]
	}

	resRow := l * n * n
	for r := 0; r < n*n; r++ {
		st.res[r] = m.aRes[l]*dot(st.nx, m.phiRes[(resRow+r)*nc:]) + m.bRes[l*n*n+r]
	}
	sinkhorn(st.res, n)

	// The lane mixing must read the pre-update stream, so results are
	// buffered before being swapped in.
	for c := 0; c < n; c++ {
		for i := 0; i < d; i++ {
			var acc float32
			for c2 := 0; c2 < n; c2++ {
				acc += st.res[c*n+c2] * st.stream[c2*d+i]
			}
			st.laneBuf[c*d+i] = acc + st.hpost[c]*st.y[i]
		}
	}
	copy(st.stream, st.laneBuf[:n*d])
}

func preOff(l, c int) float32 {
	if c == l%4 {
		return 4
	}
	return -4
}

func postOff(l, c int) float32 {
	if c == l%4 {
		return 0
	}
	return -4
}

// sinkhorn runs the 20-iteration log-domain Sinkhorn in place.
func sinkhorn(k []float32, n int) {
	lse := func(get func(i int) float32, count int) float32 {
		m := get(0)
		for i := 1; i < count; i++ {
			if v := get(i); v > m {
				m = v
			}
		}
		var s float64
		for i := 0; i < count; i++ {
			s += math.Exp(float64(get(i) - m))
		}
		return m + float32(math.Log(s))
	}
	for it := 0; it < 20; it++ {
		for r := 0; r < n; r++ {
			l := lse(func(i int) float32 { return k[r*n+i] }, n)
			for c := 0; c < n; c++ {
				k[r*n+c] -= l
			}
		}
		for c := 0; c < n; c++ {
			l := lse(func(i int) float32 { return k[i*n+c] }, n)
			for r := 0; r < n; r++ {
				k[r*n+c] -= l
			}
		}
	}
	for i := range k {
		k[i] = float32(math.Exp(float64(k[i])))
	}
}

// block computes one transformer block on st.u, writing block(u) to st.y.
func (st *runState) block(l int) {
	m := st.m
	g := m.geom
	L := m.layers[l]
	d := g.DModel

	x := make([]float32, d)
	copy(x, st.u)

	// engram injection for the site bound to this layer
	for s, site := range g.Sites {
		if site != l {
			continue
		}
		ek, ev := st.egK[s], st.egV[s]
		ru1 := make([]float32, d)
		ru2 := make([]float32, d)
		rmsUnit(x, ru1)
		rmsUnit(ek, ru2)
		alpha := sigmoid(dot(ru1, ru2) / float32(math.Sqrt(float64(d))))
		for i := range x {
			x[i] += alpha * ev[i]
		}
	}

	xn := make([]float32, d)
	zcrmsNorm(x, L.normIn, xn)

	qraw := make([]float32, m.qDim)
	kraw := make([]float32, m.kDim)
	vraw := make([]float32, m.vDim)
	gemvPar2(L.qw, xn, qraw, m.qDim, d)
	gemvPar2(L.kw, xn, kraw, m.kDim, d)
	gemvPar2(L.vw, xn, vraw, m.vDim, d)

	t := st.t
	ntaps := g.QKVConvTaps
	q := make([]float32, m.qDim)
	k := make([]float32, m.kDim)
	v := make([]float32, m.vDim)
	if g.QKVConvTaps > 0 {
		copy(st.qHist[l][(t%ntaps)*m.qDim:], qraw)
		copy(st.kHist[l][(t%ntaps)*m.kDim:], kraw)
		copy(st.vHist[l][(t%ntaps)*m.vDim:], vraw)
		for j := 0; j < ntaps; j++ {
			if t-j < 0 {
				continue
			}
			slot := (t - j) % ntaps
			qh, kh, vh := st.qHist[l][slot*m.qDim:], st.kHist[l][slot*m.kDim:], st.vHist[l][slot*m.vDim:]
			qw, kw, vw := L.qTaps[j*m.qDim:], L.kTaps[j*m.kDim:], L.vTaps[j*m.vDim:]
			for i := range q {
				q[i] += qw[i] * qh[i]
			}
			for i := range k {
				k[i] += kw[i] * kh[i]
			}
			for i := range v {
				v[i] += vw[i] * vh[i]
			}
		}
	} else {
		copy(q, qraw)
		copy(k, kraw)
		copy(v, vraw)
	}

	// per-head q/k norm + rope
	qhd := g.QKHeadDim
	for h := 0; h < g.NumHeads; h++ {
		qh := q[h*qhd : (h+1)*qhd]
		zcrmsNorm(qh, L.qNorm, qh)
		applyRope(qh, st.ropeCos[t], st.ropeSin[t])
	}
	for h := 0; h < g.NumKVHeads; h++ {
		kh := k[h*qhd : (h+1)*qhd]
		cp := make([]float32, qhd)
		copy(cp, kh)
		zcrmsNorm(cp, L.kNorm, kh)
		applyRope(kh, st.ropeCos[t], st.ropeSin[t])
	}
	copy(st.kCache[l][t*m.kDim:(t+1)*m.kDim], k)
	copy(st.vCache[l][t*m.vDim:(t+1)*m.vDim], v)

	// grouped-query causal attention with sliding window
	scale := float32(1.0 / math.Sqrt(float64(qhd)))
	global := g.globalLayer(l)
	sw := g.SlidingWindow
	repeat := g.NumHeads / g.NumKVHeads
	attnOut := make([]float32, m.outDim)
	scores := make([]float32, t+1)
	kc, vc := st.kCache[l], st.vCache[l]
	for h := 0; h < g.NumHeads; h++ {
		qh := q[h*qhd : (h+1)*qhd]
		kvh := h / repeat
		lo := 0
		if !global && sw > 0 {
			lo = t - sw + 1
			if lo < 0 {
				lo = 0
			}
		}
		n := t - lo + 1
		for s := lo; s <= t; s++ {
			scores[s-lo] = dot(qh, kc[s*m.kDim+kvh*qhd:]) * scale
		}
		softmaxInPlace(scores[:n])
		for s := lo; s <= t; s++ {
			p := scores[s-lo]
			if p == 0 {
				continue
			}
			vs := vc[s*m.vDim+kvh*g.VHeadDim:]
			o := attnOut[h*g.VHeadDim:]
			for i := 0; i < g.VHeadDim; i++ {
				o[i] += p * vs[i]
			}
		}
	}

	// gated output projection
	gate := make([]float32, d)
	gemvPar2(L.gateW, xn, gate, d, d)
	for i := range attnOut {
		attnOut[i] *= sigmoid(gate[i])
	}
	yAttn := make([]float32, d)
	gemvPar2(L.outW, attnOut, yAttn, d, d)

	zcrmsNorm(yAttn, L.postNorm, xn)
	ag := sigmoid(L.attnGate)
	for i := range x {
		x[i] += ag * xn[i]
	}

	// Monarch-Hadamard MLP
	zcrmsNorm(x, L.preHada, xn)
	mlp := st.hadamard(L, xn)
	for i := range x {
		x[i] += mlp[i]
	}
	copy(st.y, x)
}

// hadamard applies the Monarch-Hadamard MLP to xn (length d).
func (st *runState) hadamard(L *layerW, xn []float32) []float32 {
	m := st.m
	g := m.geom
	n := g.HadaN
	d := g.DModel
	ba := m.hadaBlocks()

	c8 := make([]float32, 8)
	for i, v := range xn {
		for r := 0; r < 8; r++ {
			c8[r] += v * L.condV[i*8+r]
		}
	}
	softmaxInPlace(c8)
	cond := make([]float32, n)
	for j := 0; j < n; j++ {
		var acc float32
		for r := 0; r < 8; r++ {
			acc += c8[r] * L.condU[r*n+j]
		}
		cond[j] = 1 + acc
	}

	z := make([]float32, n)
	copy(z[:d], xn)

	for i := range z {
		z[i] *= L.d1[i]
	}
	z = kronPermute(z, L.w1a, L.w1b, ba, m.hadaP1)

	for i := range z {
		v := L.d2[i]*cond[i]*z[i] + L.b2[i]
		z[i] = float32(float64(v) / (1.0 + math.Exp(-float64(v))))
	}
	z = kronPermute(z, L.w2a, L.w2b, ba, m.hadaP2)

	for i := range z {
		z[i] *= L.d3[i]
	}
	z = kronPermute(z, L.w3a, L.w3b, ba, nil)

	out := make([]float32, d)
	for i := 0; i < d; i++ {
		out[i] = L.d4[i] * z[i]
	}
	return out
}

// kronPermute computes out = (Aᵀ Z B) flattened, optionally permuted, where
// Z is the (ba×bb) reshape of z (ba == bb for a power-of-two n).
func kronPermute(z, a, b []float32, ba int, perm []float32) []float32 {
	n := ba * ba
	tmp := make([]float32, n)
	for i := 0; i < ba; i++ {
		for k := 0; k < ba; k++ {
			ak := a[i*ba+k]
			if ak == 0 {
				continue
			}
			for j := 0; j < ba; j++ {
				tmp[k*ba+j] += ak * z[i*ba+j]
			}
		}
	}
	out := make([]float32, n)
	for k := 0; k < ba; k++ {
		for l := 0; l < ba; l++ {
			var acc float32
			for j := 0; j < ba; j++ {
				acc += tmp[k*ba+j] * b[j*ba+l]
			}
			out[k*ba+l] = acc
		}
	}
	if perm != nil { // gather: result[i] = z[perm[i]]
		raw := out
		out = make([]float32, n)
		for i := range out {
			out[i] = raw[int(perm[i])]
		}
	}
	return out
}

func applyRope(x, cos, sin []float32) {
	half := len(x) / 2
	for i := 0; i < half; i++ {
		x1, x2 := x[i], x[i+half]
		c, s := cos[i], sin[i]
		x[i] = x1*c - x2*s
		x[i+half] = x2*c + x1*s
	}
}

// probePooled runs the confidence head's probe pool over all recorded cells,
// returning q*d floats (3072 for the shipped head).
func (st *runState) probePooled() []float32 {
	m := st.m
	if m.conf == nil {
		return nil
	}
	d := m.geom.DModel
	k, q := m.probeK, m.probeQ
	l1 := len(m.conf.probes) / k / d
	T := st.cellEnd()
	if T <= 0 {
		return nil
	}

	scale := float32(1.0 / math.Sqrt(float64(d)))
	scores := make([]float32, T)
	r := make([]float32, l1*k*d)
	for li := 0; li < l1; li++ {
		for kp := 0; kp < k; kp++ {
			prow := m.conf.probes[(li*k+kp)*d : (li*k+kp)*d+d]
			for tt := 0; tt < T; tt++ {
				cell := st.cells[tt*st.cellT*d+li*d : tt*st.cellT*d+li*d+d]
				scores[tt] = dot(prow, cell) * scale
			}
			softmaxInPlace(scores)
			rr := r[(li*k+kp)*d : (li*k+kp)*d+d]
			for tt := 0; tt < T; tt++ {
				wt := scores[tt]
				if wt == 0 {
					continue
				}
				cell := st.cells[tt*st.cellT*d+li*d : tt*st.cellT*d+li*d+d]
				for i := 0; i < d; i++ {
					rr[i] += wt * cell[i]
				}
			}
			rmsUnit(rr, rr)
			gr := m.conf.gain[li*k+kp]
			for i := range rr {
				rr[i] *= gr
			}
		}
	}

	u := make([]float32, l1*k)
	w := make([]float32, l1*k)
	pooled := make([]float32, q*d)
	for qi := 0; qi < q; qi++ {
		qrow := m.conf.query[qi*d:]
		for mi := 0; mi < l1*k; mi++ {
			li, kp := mi/k, mi%k
			u[mi] = dot(r[mi*d:(mi+1)*d], qrow)*scale + m.conf.rowBias[(qi*l1+li)*k+kp]
		}
		copy(w, u)
		softmaxInPlace(w)
		for mi := 0; mi < l1*k; mi++ {
			if w[mi] == 0 {
				continue
			}
			mirow := r[mi*d : (mi+1)*d]
			base := qi * d
			for i := 0; i < d; i++ {
				pooled[base+i] += w[mi] * mirow[i]
			}
		}
	}
	return pooled
}

// probeConfidence returns the raw logit of the confidence head.
func (st *runState) probeConfidence() (float32, bool) {
	m := st.m
	if m.conf == nil {
		return 0, false
	}
	pooled := st.probePooled()
	if pooled == nil {
		return 0, false
	}
	return dot(m.conf.proj, pooled) + m.conf.bias, true
}

// engramHeads derives the head count per order from geometry.
func (m *model) engramHeads() int {
	h := m.geom.NumEngramTabs / len(m.geom.Orders)
	if h < 1 {
		h = 1
	}
	return h
}

// hadaBlocks returns the Monarch block size for the power-of-two hada_n.
func (m *model) hadaBlocks() int {
	n := m.geom.HadaN
	bits := 0
	for v := n - 1; v > 0; v >>= 1 {
		bits++
	}
	return 1 << (bits / 2)
}

// gemvPar2 is the hot-path matvec.
func gemvPar2(w, x, y []float32, rows, cols int) {
	if rows*cols >= 1<<13 && runtime.NumCPU() > 1 {
		gemv(w, x, y, rows, cols)
		return
	}
	gemvRange(w, x, y, 0, rows, cols)
}
