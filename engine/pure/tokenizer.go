package pure

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// Token type codes from the .cact RAW tokenizer dump.
const (
	tkNormal     = 0
	tkUnknown    = 1
	tkControl    = 2
	tkUserDef    = 3
	tkByte       = 4
	spMetaSpace  = "▁"
	tokenizerHDR = 24 // 5 u32 + u8 + u8 + u16
	tokenizerREC = 7  // f32 + u8 + u16 (packed, no alignment)
)

type tokenizer struct {
	pieces    []string
	scores    []float32
	types     []byte
	padID     int
	eosID     int
	bosID     int
	unkID     int
	addDummy  bool
	byteFallb bool

	p2id    map[string]int
	byteID  map[byte]int
	markers []string // user-defined markers, longest first
}

func newTokenizer(blob []byte) (*tokenizer, error) {
	if len(blob) < tokenizerHDR {
		return nil, fmt.Errorf("pure: tokenizer blob too small")
	}
	t := &tokenizer{}
	t.pieces = nil
	n := int(binary.LittleEndian.Uint32(blob[0:]))
	t.padID = int(binary.LittleEndian.Uint32(blob[4:]))
	t.eosID = int(binary.LittleEndian.Uint32(blob[8:]))
	t.bosID = int(binary.LittleEndian.Uint32(blob[12:]))
	t.unkID = int(binary.LittleEndian.Uint32(blob[16:]))
	t.addDummy = blob[20] != 0
	t.byteFallb = blob[21] != 0

	off := tokenizerHDR
	t.pieces = make([]string, n)
	t.scores = make([]float32, n)
	t.types = make([]byte, n)
	t.p2id = make(map[string]int, n)
	t.byteID = make(map[byte]int, 256)
	for i := 0; i < n; i++ {
		if off+tokenizerREC > len(blob) {
			return nil, fmt.Errorf("pure: tokenizer record %d overruns blob", i)
		}
		t.scores[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[off:]))
		typ := blob[off+4]
		ln := int(binary.LittleEndian.Uint16(blob[off+5:]))
		off += tokenizerREC
		if off+ln > len(blob) {
			return nil, fmt.Errorf("pure: tokenizer surface %d overruns blob", i)
		}
		piece := string(blob[off : off+ln])
		off += ln
		t.pieces[i] = piece
		t.types[i] = typ
		t.p2id[piece] = i
		if typ == tkByte && len(piece) == 6 { // "<0xNN>"
			b, err := hexByte(piece[3], piece[4])
			if err == nil {
				t.byteID[b] = i
			}
		}
	}
	for i, typ := range t.types {
		if typ == tkUserDef {
			t.markers = append(t.markers, t.pieces[i])
		}
	}
	sort.SliceStable(t.markers, func(a, b int) bool {
		return len(t.markers[a]) > len(t.markers[b])
	})
	return t, nil
}

func hexByte(hi, lo byte) (byte, error) {
	f := func(c byte) (int, error) {
		switch {
		case '0' <= c && c <= '9':
			return int(c - '0'), nil
		case 'a' <= c && c <= 'f':
			return int(c-'a') + 10, nil
		case 'A' <= c && c <= 'F':
			return int(c-'A') + 10, nil
		}
		return 0, fmt.Errorf("bad hex digit %q", c)
	}
	h, err := f(hi)
	if err != nil {
		return 0, err
	}
	l, err := f(lo)
	if err != nil {
		return 0, err
	}
	return byte(h<<4 | l), nil
}

// encode mirrors the RefTokenizer reference encoder in export.py.
func (t *tokenizer) encode(text string) []int {
	if text == "" {
		return nil
	}
	esc := strings.ReplaceAll(text, " ", spMetaSpace)
	if t.addDummy {
		esc = spMetaSpace + esc
	}
	var ids []int
	var buf []rune
	for i := 0; i < len(esc); {
		matched := ""
		for _, m := range t.markers {
			if strings.HasPrefix(esc[i:], m) {
				matched = m
				break
			}
		}
		if matched != "" {
			ids = append(ids, t.bpe(buf)...)
			buf = buf[:0]
			ids = append(ids, t.p2id[matched])
			i += len(matched)
		} else {
			r, size := utf8.DecodeRuneInString(esc[i:])
			buf = append(buf, r)
			i += size
		}
	}
	ids = append(ids, t.bpe(buf)...)
	return ids
}

// bpe runs greedy longest-score pair merges over a rune segment.
func (t *tokenizer) bpe(seg []rune) []int {
	if len(seg) == 0 {
		return nil
	}
	syms := make([]string, len(seg))
	for i, r := range seg {
		syms[i] = string(r)
	}
	for len(syms) > 1 {
		bestScore := float32(0)
		bestJ := -1
		found := false
		for j := 0; j < len(syms)-1; j++ {
			idx, ok := t.p2id[syms[j]+syms[j+1]]
			if ok && (!found || t.scores[idx] > bestScore) {
				bestScore, bestJ, found = t.scores[idx], j, true
			}
		}
		if bestJ < 0 {
			break
		}
		syms[bestJ] = syms[bestJ] + syms[bestJ+1]
		syms = append(syms[:bestJ+1], syms[bestJ+2:]...)
	}
	ids := make([]int, 0, len(syms))
	for _, s := range syms {
		if idx, ok := t.p2id[s]; ok {
			ids = append(ids, idx)
		} else if t.byteFallb {
			for b := 0; b < len(s); b++ {
				if id, ok := t.byteID[s[b]]; ok {
					ids = append(ids, id)
				} else {
					ids = append(ids, t.unkID)
				}
			}
		} else {
			ids = append(ids, t.unkID)
		}
	}
	return ids
}

// decode maps token ids back to text.
func (t *tokenizer) decode(ids []int) string {
	var sb strings.Builder
	var pending []byte
	flush := func() {
		if len(pending) > 0 {
			sb.Write(pending)
			pending = pending[:0]
		}
	}
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			continue
		}
		switch t.types[id] {
		case tkByte:
			piece := t.pieces[id]
			if len(piece) == 6 {
				if b, err := hexByte(piece[3], piece[4]); err == nil {
					pending = append(pending, b)
				}
			}
		case tkControl, tkUnknown:
			flush()
		default:
			flush()
			sb.WriteString(t.pieces[id])
		}
	}
	flush()
	text := strings.ReplaceAll(sb.String(), spMetaSpace, " ")
	if t.addDummy && strings.HasPrefix(text, " ") {
		text = text[1:]
	}
	return text
}
