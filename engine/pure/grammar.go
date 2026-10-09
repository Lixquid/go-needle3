package pure

import "strings"

// callGrammar is a minimal JSON validator for the <tool_call> region that
// mirrors the engine's schema constraints: tool names come from the loaded
// set, object keys from the schema ("name"/"arguments" then the tool's
// properties), and the array must close before </tool_call>.
type callGrammar struct {
	toolNames  map[string]bool
	schemaKeys map[string]map[string]bool

	st         int
	stack      []byte // 'a' array, 'o' object
	key        []byte
	inKey      bool
	inStr      bool
	escaped    bool
	escaping   bool
	strIsName  bool // the string being read is the "name" value
	curKey     string
	pendingKey bool // key closed, waiting for ':'
	toolName   string
	nameBuf    []byte
	numOpen    bool
	litLeft    int // remaining chars of true/false/null
	done       bool
	err        bool
}

const (
	gAfterCall = iota
	gArrElem
	gObjKey
	gObjKeyStr // reading key chars
	gColon     // after key, waiting ':'
	gValueStart
	gStr        // reading value string
	gNum        // reading number chars
	gLiteral    // reading true/false/null tail
	gAfterValue // after complete value: ',' or close
	gClosed
)

func newCallGrammar(tools map[string]map[string]any, names map[string]bool) *callGrammar {
	g := &callGrammar{toolNames: names, schemaKeys: map[string]map[string]bool{}}
	for n, p := range tools {
		keys := map[string]bool{}
		if props, ok := p["properties"].(map[string]any); ok {
			for k := range props {
				keys[k] = true
			}
		}
		g.schemaKeys[n] = keys
	}
	g.st = gAfterCall
	return g
}

func (g *callGrammar) depth() int { return len(g.stack) }

// feed advances the parser over one token's decoded text.
func (g *callGrammar) feed(s string) {
	if g.err || g.done {
		g.err = true
		return
	}
	for i := 0; i < len(s) && !g.err && !g.done; i++ {
		c := s[i]
		switch {
		case g.inStr:
			if g.escaped {
				g.escaped = false
				continue
			}
			if c == '\\' {
				g.escaped = true
				continue
			}
			if c == '"' {
				g.inStr = false
				g.endString()
			} else if g.strIsName {
				g.nameBuf = append(g.nameBuf, c)
				if !g.namePrefixOK() {
					g.err = true
					return
				}
			}
		case g.inKey:
			if c == '"' {
				g.inKey = false
				g.endKey()
			} else {
				g.key = append(g.key, c)
				if !g.keyPrefixOK() {
					g.err = true
					return
				}
			}
		case g.numOpen:
			if (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
				continue
			}
			g.numOpen = false
			g.transition(c)
		case g.litLeft > 0:
			if c >= 'a' && c <= 'z' {
				g.litLeft--
				if g.litLeft == 0 {
					g.st = gAfterValue
				}
				continue
			}
			g.err = true
		default:
			g.transition(c)
		}
	}
	// trailing states that are valid prefixes: strings/numbers still open
}

// transition handles a non-string character in structural states.
func (g *callGrammar) transition(c byte) {
	if c == ' ' || c == '\n' || c == '\t' || c == '\r' {
		return
	}
	switch g.st {
	case gAfterCall:
		if c == '[' {
			g.stack = append(g.stack, 'a')
			g.st = gArrElem
			return
		}
		g.err = true
	case gArrElem:
		switch c {
		case '{':
			g.stack = append(g.stack, 'o')
			g.st = gObjKey
			return
		case ']':
			if !g.pop() {
				return
			}
			if g.depth() == 0 {
				g.st = gClosed
				g.done = true
				return
			}
			g.st = gAfterValue
			return
		}
		g.err = true
	case gObjKey:
		switch c {
		case '"':
			g.inKey = true
			g.key = g.key[:0]
			return
		case '}':
			if !g.pop() {
				return
			}
			g.st = gAfterValue
			return
		}
		g.err = true
	case gColon:
		if c == ':' {
			g.st = gValueStart
			return
		}
		g.err = true
	case gValueStart:
		switch c {
		case '"':
			g.inStr = true
			g.strIsName = g.curKey == "name" && g.depth() == 2
			if g.strIsName {
				g.nameBuf = g.nameBuf[:0]
			}
			g.st = gStr
			return
		case '{':
			if g.curKey != "arguments" || g.depth() != 2 {
				g.err = true
				return
			}
			g.stack = append(g.stack, 'o')
			g.st = gObjKey
			return
		case '[':
			g.err = true // array values unsupported in needle schemas
		case 't':
			g.litLeft = 3 // rue
			return
		case 'f':
			g.litLeft = 4 // alse
			return
		case 'n':
			g.litLeft = 3 // ull
			return
		default:
			if c == '-' || (c >= '0' && c <= '9') {
				g.numOpen = true
				g.st = gNum
				return
			}
		}
		g.err = true
	case gClosed:
		if c == ' ' || c == '\n' || c == '\t' || c == '\r' {
			return
		}
		g.err = true
	case gAfterValue, gNum:
		switch c {
		case ',':
			if g.top() == 'a' {
				g.st = gArrElem2
			} else {
				g.st = gObjKey
			}
			return
		case '}':
			if !g.pop() {
				return
			}
			g.st = gAfterValue
			return
		case ']':
			if g.top() != 'a' {
				g.err = true
				return
			}
			if !g.pop() {
				return
			}
			if g.depth() == 0 {
				g.st = gClosed
				g.done = true
				return
			}
			g.st = gAfterValue
			return
		}
		g.err = true
	default:
		g.err = true
	}
}

// extra states folded via gAfterValue semantics:
const (
	gArrElem2 = gArrElem // after ',' inside the array: expect '{' or ']'
	gArrClose = gAfterValue
)

func (g *callGrammar) pop() bool {
	if len(g.stack) == 0 {
		g.err = true
		return false
	}
	g.stack = g.stack[:len(g.stack)-1]
	return true
}

func (g *callGrammar) top() byte {
	if len(g.stack) == 0 {
		return 0
	}
	return g.stack[len(g.stack)-1]
}

func (g *callGrammar) allowedKeys() []string {
	if g.depth() == 2 {
		return []string{"name", "arguments"}
	}
	ks := g.schemaKeys[g.toolName]
	out := make([]string, 0, len(ks))
	for k := range ks {
		out = append(out, k)
	}
	return out
}

func (g *callGrammar) keyPrefixOK() bool {
	p := string(g.key)
	for _, k := range g.allowedKeys() {
		if strings.HasPrefix(k, p) {
			return true
		}
	}
	return false
}

func (g *callGrammar) namePrefixOK() bool {
	p := string(g.nameBuf)
	for k := range g.toolNames {
		if strings.HasPrefix(k, p) {
			return true
		}
	}
	return false
}

func (g *callGrammar) endKey() {
	k := string(g.key)
	switch g.depth() {
	case 2: // tool object level
		if k != "name" && k != "arguments" {
			g.err = true
			return
		}
	case 3: // arguments level
		tn := g.toolName
		if !g.schemaKeys[tn][k] {
			g.err = true
			return
		}
	default:
		g.err = true
		return
	}
	g.curKey = k
	g.st = gColon
}

func (g *callGrammar) endString() {
	if g.strIsName {
		tn := string(g.nameBuf)
		if !g.toolNames[tn] {
			g.err = true
		}
		g.toolName = tn
		g.strIsName = false
	}
	g.st = gAfterValue
}

func (g *callGrammar) clone() *callGrammar {
	c := *g
	c.stack = append([]byte(nil), g.stack...)
	c.key = append([]byte(nil), g.key...)
	c.nameBuf = append([]byte(nil), g.nameBuf...)
	return &c
}
