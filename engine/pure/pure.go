// Package pure is a pure-Go port of the Needle 3 runtime: it parses the
// .cact container, dequantizes CQ weights, runs the Simple Attention Network
// forward pass, and decodes greedily. No cgo, no libc.
//
// It mirrors the engine/cgo API (both implement needle.Engine), but the
// numerics are float32: the shipped C++ engine runs int8 activations and an
// int8 KV cache plus grammar-constrained decoding, so rare token choices can
// differ. The tool-call JSON is expected to agree on the demo presets.
package pure

import (
	"encoding/json"
	"errors"
	"math"
	"regexp"
	"strings"

	"github.com/Zetaphor/needle-go"
)

// Token ids fixed by the vocabulary.
const (
	TokPad         = 0
	TokEOS         = 1
	TokBOS         = 2
	TokIMStart     = 4
	TokIMEnd       = 5
	TokThinkStart  = 6
	TokThinkEnd    = 7
	TokToolsStart  = 8
	TokToolsEnd    = 9
	TokCallStart   = 10
	TokCallEnd     = 11
	TokResultStart = 12
	TokResultEnd   = 13
)

// Agent is a pure-Go Needle 3 agent bound to one toolset.
type Agent struct {
	m       *model
	opts    needle.Options
	prefix  []int
	schemas map[string]map[string]any
	names   map[string]bool

	st          *runState
	pendingCall bool
	closed      bool
}

var _ needle.Engine = (*Agent)(nil)

// New dequantizes the model (about a second) and prepares the toolset.
func New(opts needle.Options) (*Agent, error) {
	if len(opts.Weights) == 0 {
		return nil, errors.New("pure: Options.Weights is empty (pass the .cact model bytes)")
	}
	opts = opts.WithDefaults()
	m, err := loadModel(opts.Weights)
	if err != nil {
		return nil, err
	}
	a := &Agent{m: m, opts: opts}

	toolsJSON, err := opts.ToolsJSON()
	if err != nil {
		return nil, err
	}
	var sb strings.Builder
	sb.WriteString("<|im_start|>")
	if opts.System != "" {
		sb.WriteString("system\n" + opts.System + "<|im_end|>\n<|im_start|>")
	}
	sb.WriteString("user\n<tools>" + string(toolsJSON) + "</tools>")
	a.prefix = append([]int{TokBOS}, m.tok.encode(sb.String())...)
	a.schemas = make(map[string]map[string]any, len(opts.Tools))
	a.names = make(map[string]bool, len(opts.Tools))
	for _, t := range opts.Tools {
		a.names[t.Name] = true
		a.schemas[t.Name] = t.Parameters
	}
	return a, nil
}

// Complete answers one turn.
func (a *Agent) Complete(text string) (*needle.Response, error) {
	if a.closed {
		return nil, errors.New("pure: agent is closed")
	}
	if a.st == nil {
		a.st = newRunState(a.m, len(a.prefix)+a.opts.MaxNewTokens+32)
		for _, id := range a.prefix {
			if _, err := a.feed(id); err != nil {
				return nil, err
			}
		}
	}

	var turn string
	if a.pendingCall {
		turn = "\n<|im_start|>tool\n<tool_result>" + text + "</tool_result><|im_end|>\n<|im_start|>assistant\n"
	} else {
		turn = "\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
	}

	var logits []float32
	var err error
	for _, id := range a.m.tok.encode(turn) {
		if logits, err = a.feed(id); err != nil {
			return nil, err
		}
	}
	// The engine forces <think> as the first token of every turn.
	if logits, err = a.feed(TokThinkStart); err != nil {
		return nil, err
	}

	// Greedy decode with a schema-constrained JSON region after <tool_call>,
	// mirroring the engine's constrained generation.
	generated := []int{TokThinkStart}
	logProb, nTok := 0.0, 1
	var gram *callGrammar
	for i := 0; i < a.opts.MaxNewTokens; i++ {
		if gram != nil && !gram.err {
			maskCall(logits, gram, a.m.tok)
		}
		next, p := argmax(logits)
		generated = append(generated, next)
		if p < 1e-9 {
			p = 1e-9
		}
		logProb += math.Log(p)
		nTok++
		if next == TokEOS || next == TokCallEnd {
			break
		}
		if next == TokCallStart {
			gram = newCallGrammar(a.schemas, a.names)
		} else if gram != nil {
			gram.feed(pieceText(a.m.tok, next))
		}
		if logits, err = a.feed(next); err != nil {
			return nil, err
		}
	}

	return a.assembleResponse(generated, math.Exp(logProb/float64(nTok)))
}

func (a *Agent) feed(id int) ([]float32, error) {
	if a.st.t+1 >= a.m.geom.MaxSeqLen {
		return nil, errors.New("pure: context window full")
	}
	return a.st.step(id), nil
}

// Reset rewinds the conversation, keeping the tools loaded.
func (a *Agent) Reset() error {
	a.st = nil
	a.pendingCall = false
	return nil
}

// Run executes the full agentic loop using Tool.Fn handlers.
func (a *Agent) Run(query string) (*needle.Response, error) {
	return needle.RunLoopFunc(a.Complete, a.opts.Tools, query, a.opts.MaxSteps)
}

// Embed returns the probe-pool embedding of a text (3072 floats, unit norm).
// It runs an independent forward pass and does not disturb the conversation.
func (a *Agent) Embed(text string) ([]float32, error) {
	if a.closed {
		return nil, errors.New("pure: agent is closed")
	}
	st := newRunState(a.m, 64)
	ids := append([]int{TokBOS}, a.m.tok.encode(text)...)
	for _, id := range ids {
		st.step(id)
	}
	pooled := st.probePooled()
	if pooled == nil {
		return nil, errors.New("pure: no confidence head for embeddings")
	}
	var norm float64
	for _, v := range pooled {
		norm += float64(v) * float64(v)
	}
	norm = 1 / (math.Sqrt(norm) + 1e-12)
	for i := range pooled {
		pooled[i] *= float32(norm)
	}
	return pooled, nil
}

// Close releases the agent.
func (a *Agent) Close() error {
	if a.closed {
		return nil
	}
	a.closed = true
	a.st = nil
	return nil
}

var (
	thinkRe = regexp.MustCompile(`(?s)<think>(.*?)</think>`)
	callRe  = regexp.MustCompile(`(?s)<tool_call>(.*?)</tool_call>`)
)

func (a *Agent) assembleResponse(generated []int, decodeP float64) (*needle.Response, error) {
	text := a.m.tok.decode(generated)
	r := &needle.Response{Type: "respond", Success: true, Raw: text}

	if mm := thinkRe.FindStringSubmatch(text); mm != nil {
		r.Reasoning = strings.TrimSpace(mm[1])
	}
	if mm := callRe.FindStringSubmatch(text); mm != nil {
		r.Type = "call"
		var calls []struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		payload := strings.TrimSpace(mm[1])
		if payload == "" {
			payload = "[]"
		}
		if err := json.Unmarshal([]byte(payload), &calls); err != nil {
			r.Success = false
			r.Error = "invalid tool_call JSON: " + truncate(payload, 120)
			r.ErrorCode = "parse"
		}
		for _, c := range calls {
			if c.Arguments == nil {
				c.Arguments = map[string]any{}
			}
			r.FunctionCalls = append(r.FunctionCalls, needle.Call{Name: c.Name, Arguments: c.Arguments})
		}
	}
	// Decode confidence: geometric mean of greedy token probabilities,
	// floored by the calibrated probe head.
	conf := decodeP
	if logit, ok := a.st.probeConfidence(); ok {
		if p := float64(sigmoid(logit)); p < conf {
			conf = p
		}
	}
	r.Confidence = conf
	r.HasConfidence = true

	// Confidence gate, mirroring the reference engine: low-confidence calls
	// are withheld (FunctionCalls emptied, kept in SuppressedCalls). The
	// 0.1 threshold is the value observed via the differential fuzz corpus
	// (suppressed at conf <= 0.093, emitted at >= 0.116 in all 35 cases).
	const suppressThreshold = 0.1
	if len(r.FunctionCalls) > 0 && conf < suppressThreshold {
		r.SuppressedCalls = r.FunctionCalls
		r.FunctionCalls = nil
	}
	a.pendingCall = r.Type == "call" && len(r.FunctionCalls) > 0
	return r, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// argmax returns the winning token and its softmax probability.
func argmax(logits []float32) (int, float64) {
	best, maxv := 0, logits[0]
	for i, v := range logits[1:] {
		if v > maxv {
			maxv, best = v, i+1
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - maxv))
	}
	return best, 1.0 / sum
}

// pieceText renders a token's literal text with ▁ mapped back to a space.
func pieceText(tok *tokenizer, id int) string {
	if tok.types[id] == tkByte {
		p := tok.pieces[id] // "<0xNN>"
		var b byte
		for _, c := range p[3:5] {
			b <<= 4
			switch {
			case c >= '0' && c <= '9':
				b |= byte(c - '0')
			case c >= 'a' && c <= 'f':
				b |= byte(c-'a') + 10
			case c >= 'A' && c <= 'F':
				b |= byte(c-'A') + 10
			}
		}
		return string([]byte{b})
	}
	return strings.ReplaceAll(tok.pieces[id], spMetaSpace, " ")
}

// maskCall restricts logits to tokens that keep the tool-call JSON legal.
func maskCall(logits []float32, g *callGrammar, tok *tokenizer) {
	const neg = float32(-1e30)
	for id := range logits {
		var legal bool
		switch {
		case id == TokCallEnd || id == TokEOS:
			legal = g.done
		case tok.types[id] == tkControl || tok.types[id] == tkUserDef || tok.types[id] == tkUnknown:
			legal = false
		case g.done:
			legal = false
		default:
			c := g.clone()
			c.feed(pieceText(tok, id))
			legal = !c.err
		}
		if !legal {
			logits[id] = neg
		}
	}
}
