package pure

import (
	"encoding/binary"
	"fmt"
	"math"
)

// cact container constants (needle3 tag).
const (
	cactTag    = 0x05E12A84
	cactTagN2  = 0x05E12A83
	cactRecLen = 44
	cactAlign  = 64
)

// dtype codes.
const (
	dtFP16 = 1
	dtFP32 = 2
	dtCQ   = 3
	dtRAW  = 4
)

// geometry mirrors the .cact header.
type geometry struct {
	NumTensors  int
	CodebookLen int
	KVWindow    int
	KVBits      int

	Vocab      int
	OutVocab   int
	DModel     int
	NumHeads   int
	NumKVHeads int
	NumLayers  int
	QKHeadDim  int
	VHeadDim   int
	MaxSeqLen  int
	HadaN      int
	MHCLanes   int

	SlidingWindow  int
	GlobalMaskLo   uint32
	GlobalMaskHi   uint32
	QKVConvTaps    int
	EngramSlots    int
	EngramSubDim   int
	NumEngramTabs  int
	EngramConvTaps int
	EngramDilation int
	EngramSeedHds  int
	Orders         []int
	Sites          []int
	RopeTheta      float32
}

func (g geometry) globalLayer(l int) bool {
	if l < 32 {
		return g.GlobalMaskLo>>uint(l)&1 == 1
	}
	return g.GlobalMaskHi>>uint(l-32)&1 == 1
}

type dirRecord struct {
	dtype  byte
	ndim   byte
	shape  [4]uint32
	offset uint64
	nbytes uint64
	group  uint32
	bits   uint32
}

func (r dirRecord) numel() int {
	n := 1
	for i := 0; i < int(r.ndim); i++ {
		n *= int(r.shape[i])
	}
	return n
}

// parseHeader reads geometry + codebook + directory from a .cact image.
func parseHeader(data []byte) (geometry, []float32, []dirRecord, error) {
	if len(data) < 49*4 {
		return geometry{}, nil, nil, fmt.Errorf("pure: file too small (%d bytes)", len(data))
	}
	g := geometry{}
	h := func(i int) uint32 { return binary.LittleEndian.Uint32(data[4*i:]) }
	tag := h(0)
	if tag == cactTagN2 {
		return g, nil, nil, fmt.Errorf("pure: Needle 2 archive (tag 0x%08X), Needle 3 required", tag)
	}
	if tag != cactTag {
		return g, nil, nil, fmt.Errorf("pure: bad .cact tag 0x%08X", tag)
	}
	g.NumTensors = int(h(1))
	g.CodebookLen = int(h(2))
	g.KVWindow = int(h(3))
	g.KVBits = int(h(4))
	g.Vocab = int(h(5))
	g.OutVocab = int(h(6))
	g.DModel = int(h(7))
	g.NumHeads = int(h(8))
	g.NumKVHeads = int(h(9))
	g.NumLayers = int(h(10))
	g.QKHeadDim = int(h(11))
	g.VHeadDim = int(h(12))
	g.MaxSeqLen = int(h(13))
	g.HadaN = int(h(14))
	g.MHCLanes = int(h(15))
	g.SlidingWindow = int(h(16))
	g.GlobalMaskLo = h(17)
	g.GlobalMaskHi = h(18)
	g.QKVConvTaps = int(h(19))
	g.EngramSlots = int(h(20))
	g.EngramSubDim = int(h(21))
	g.NumEngramTabs = int(h(22))
	g.EngramConvTaps = int(h(23))
	g.EngramDilation = int(h(24))
	g.EngramSeedHds = int(h(25))
	nOrders := int(h(26))
	if nOrders < 0 || nOrders > 4 {
		return g, nil, nil, fmt.Errorf("pure: bad engram order count %d", nOrders)
	}
	for i := 0; i < nOrders; i++ {
		g.Orders = append(g.Orders, int(h(27+i)))
	}
	nSites := int(h(31))
	if nSites < 0 || nSites > 16 {
		return g, nil, nil, fmt.Errorf("pure: bad engram site count %d", nSites)
	}
	for i := 0; i < nSites; i++ {
		g.Sites = append(g.Sites, int(h(32+i)))
	}
	g.RopeTheta = math.Float32frombits(h(48))

	off := 49 * 4
	if off+g.CodebookLen*4 > len(data) {
		return g, nil, nil, fmt.Errorf("pure: codebook overruns file")
	}
	codebook := make([]float32, g.CodebookLen)
	for i := range codebook {
		codebook[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off+4*i:]))
	}
	off += g.CodebookLen * 4

	need := g.NumTensors * cactRecLen
	if off+need > len(data) {
		return g, nil, nil, fmt.Errorf("pure: directory overruns file")
	}
	recs := make([]dirRecord, g.NumTensors)
	for i := range recs {
		r := data[off+i*cactRecLen:]
		recs[i].dtype = r[0]
		recs[i].ndim = r[1]
		for j := 0; j < 4; j++ {
			recs[i].shape[j] = binary.LittleEndian.Uint32(r[4+4*j:])
		}
		recs[i].offset = binary.LittleEndian.Uint64(r[20:])
		recs[i].nbytes = binary.LittleEndian.Uint64(r[28:])
		recs[i].group = binary.LittleEndian.Uint32(r[36:])
		recs[i].bits = binary.LittleEndian.Uint32(r[40:])
	}
	return g, codebook, recs, nil
}

// fp16 → float32
func f16(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits & 0x3FF)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case exp == 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

func decodeFP16(blob []byte) []float32 {
	out := make([]float32, len(blob)/2)
	for i := range out {
		out[i] = f16(binary.LittleEndian.Uint16(blob[2*i:]))
	}
	return out
}

func decodeFP32(blob []byte) []float32 {
	out := make([]float32, len(blob)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[4*i:]))
	}
	return out
}

// codebookFor returns the unit-sphere codebook for a bit width, already
// divided by sqrt(group) by the exporter.
func codebookFor(codebook []float32, bits uint32) ([]float32, error) {
	levels := int(bits) == 3 // widths: 2, 3, 4 → counts 4, 8, 16
	_ = levels
	start, count := 0, 0
	switch bits {
	case 2:
		start, count = 0, 4
	case 3:
		start, count = 4, 8
	case 4:
		start, count = 12, 16
	default:
		return nil, fmt.Errorf("pure: unsupported CQ width %d", bits)
	}
	if start+count > len(codebook) {
		return nil, fmt.Errorf("pure: codebook too short")
	}
	return codebook[start : start+count], nil
}

// fwhtNormalized applies the in-place Walsh-Hadamard transform on n=power of 2,
// scaled by 1/sqrt(n) — the matrix H the container promises.
func fwhtNormalized(a []float32) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j] = x + y
				a[j+h] = x - y
			}
		}
	}
	s := float32(1.0 / math.Sqrt(float64(n)))
	for i := range a {
		a[i] *= s
	}
}

// dequantCQ expands a CQ blob to a row-major [out][in] float32 matrix.
// Blob: packed indices (LSB-first bitstream per row, chunks of 8 indices
// emitted as `bits` little-endian bytes), then out*(in_pad/group) FP16 norms.
func dequantCQ(blob []byte, out, inDim int, bits, group uint32, codebook []float32) ([]float32, error) {
	if bits != 2 && bits != 3 && bits != 4 {
		return nil, fmt.Errorf("pure: unsupported record bits %d", bits)
	}
	cb, err := codebookFor(codebook, bits)
	if err != nil {
		return nil, err
	}
	g := int(group)
	if g <= 0 {
		g = 128
	}
	inPad := (inDim + g - 1) / g * g
	rowBytes := inPad * int(bits) / 8
	need := out*rowBytes + out*(inPad/g)*2
	if len(blob) < need {
		return nil, fmt.Errorf("pure: CQ blob short: %d < %d", len(blob), need)
	}
	idxBytes := blob[:out*rowBytes]
	normsOff := out * rowBytes

	mat := make([]float32, out*inDim)
	buf := make([]float32, g)
	for r := 0; r < out; r++ {
		rowBitsStart := r * rowBytes * 8
		for gi := 0; gi < inPad/g; gi++ {
			base := gi * g
			for k := 0; k < g; k++ {
				p := rowBitsStart + (base+k)*int(bits)
				// read `bits` bits LSB-first, possibly spanning bytes
				v := uint32(idxBytes[p/8]) >> uint(p%8)
				if p%8+int(bits) > 8 {
					v |= uint32(idxBytes[p/8+1]) << uint(8-p%8)
				}
				buf[k] = cb[v&((1<<bits)-1)]
			}
			// dequant: (cb * norm) @ H
			norm := f16(binary.LittleEndian.Uint16(blob[normsOff+r*(inPad/g)*2+gi*2:]))
			for k := range buf {
				buf[k] *= norm
			}
			fwhtNormalized(buf)
			end := base + g
			if end > inDim {
				end = inDim
			}
			copy(mat[r*inDim+base:r*inDim+end], buf[:end-base])
		}
	}
	return mat, nil
}
