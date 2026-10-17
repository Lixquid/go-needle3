package needle

// Strict extraction grounding, ported from the Python package's
// _source_years / _walk_grounding / _validate_extraction.
//
// Two sources of doubt are checked:
//
//  1. Temporal drift. A "date"/"date-time" value may only carry a year that is
//     written in the input, plus the system date's year when the input reasons
//     relatively ("due tomorrow"). The model invents plausible years freely;
//     this is the check that catches it.
//  2. Engine-reported flags. The C engine returns a "validation" object with
//     the argument paths it considers ungrounded, and a negation marker. A
//     flagged number is cleared when the same number is written in the input —
//     which is what lets "$1,200.00" satisfy a total of 1200 without opening
//     the door to a fabricated value.
//
// Go's regexp is RE2: no lookahead, no lookbehind. Every guard the Python
// patterns write as (?…) is re-implemented here as an index check on the text
// around a match, or as an explicit scanner (numbers), so the two ports agree.

import (
	"encoding/json"
	"errors"
	"math/big"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// monthPattern matches a month name, abbreviated or spelled out.
const monthPattern = `(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|` +
	`aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)`

// yearPattern is one way a year is written in the wild. The capture is a
// greedy \d+ run whose length is then checked against min/max, because RE2
// resolves a bounded repeat such as \d{3,4} into whichever submatch it likes;
// timeGuard rejects the number being the hour of a time ("5 June 19:30",
// "June 5 7 pm"), which no ISO date argument can carry.
type yearPattern struct {
	re        *regexp.Regexp
	min       int
	max       int
	timeGuard bool
}

// yearPatterns find the years literally written in a text. A number after a
// day and a month is its year unless it is a time; and a day- or month-first
// numeric date ("5/6/24") must not mint its leading component, since no ISO
// argument can ever match it.
var yearPatterns = []yearPattern{
	{regexp.MustCompile(`\b\d{1,2}(?:st|nd|rd|th)?\s+` + monthPattern + `[\s,]+(\d+)`), 1, 4, true},
	{regexp.MustCompile(`\b` + monthPattern + `\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s+(\d+)`), 1, 4, true},
	{regexp.MustCompile(`\b` + monthPattern + `[\s,]+(\d+)`), 3, 4, false},
	{regexp.MustCompile(`\byear\s+(\d+)`), 1, 4, false},
}

// isoDateRe is the year-first numeric date ("2024-03-15", "2024/03/15").
var isoDateRe = regexp.MustCompile(`\d{4}[-/]\d{1,2}[-/]\d{1,2}`)

var relativeCueRe = regexp.MustCompile(`(?i)\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`)

var (
	isoStampRe   = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)
	dateKeyRe    = regexp.MustCompile(`(^|[^A-Za-z])date:`)
	yearPrefixRe = regexp.MustCompile(`^(\d{4})-`)
	ampmRe       = regexp.MustCompile(`^[ap]\.?m([^a-z0-9]|$)`)
)

func isDigit(b byte) bool { return b >= '0' && b <= '9' }
func isAlnum(b byte) bool {
	return b >= '0' && b <= '9' || b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z'
}
func isWordByte(b byte) bool {
	return isAlnum(b) || b == '_' || b == '.' || b == ','
}

// afterYear reports whether a year candidate may stand, checking the text that
// follows it: it must not run into another word character, a clock ("19:30")
// or an am/pm marker.
func afterYear(text string, end int, timeGuard bool) bool {
	if end >= len(text) {
		return true
	}
	if isAlnum(text[end]) {
		return false
	}
	if timeGuard {
		if text[end] == ':' && end+1 < len(text) && isDigit(text[end+1]) {
			return false
		}
		rest := strings.TrimLeft(text[end:], " ")
		if ampmRe.MatchString(rest) {
			return false
		}
	}
	return true
}

// sourceYears returns the years written in a text.
func sourceYears(text string) map[int64]bool {
	years := map[int64]bool{}
	lowered := strings.ToLower(text)
	add := func(s string) {
		if y, err := strconv.ParseInt(s, 10, 64); err == nil {
			years[y] = true
		}
	}
	for _, p := range yearPatterns {
		for _, m := range p.re.FindAllStringSubmatchIndex(lowered, -1) {
			digits := lowered[m[2]:m[3]]
			if len(digits) < p.min || len(digits) > p.max {
				continue
			}
			if !afterYear(lowered, m[3], p.timeGuard) {
				continue
			}
			add(digits)
		}
	}
	for _, m := range isoDateRe.FindAllStringIndex(lowered, -1) {
		if m[0] > 0 && isDigit(lowered[m[0]-1]) {
			continue // not the start of the year ("12024-05-06")
		}
		if m[1] < len(lowered) && isDigit(lowered[m[1]]) {
			continue // the day ran into more digits
		}
		add(lowered[m[0] : m[0]+4])
	}
	return years
}

// relativeCue reports whether a text reasons about time relative to now.
func relativeCue(text string) bool { return relativeCueRe.MatchString(text) }

// licensedYears are the years a date argument may carry: the ones written in
// the input, plus the system date's years when the input reasons relatively.
func licensedYears(seen map[int64]bool, system string, relative bool) map[int64]bool {
	years := make(map[int64]bool, len(seen))
	for y := range seen {
		years[y] = true
	}
	if len(years) > 0 && system != "" && relative {
		for y := range sourceYears(system) {
			years[y] = true
		}
	}
	return years
}

// WithDateFact prefixes the local date fact unless the caller already supplied
// one, so a relative date has something to be relative to. A system prompt
// that already carries a "date:" key, or any ISO stamp, is left alone; JSON
// facts (an object) are passed through untouched.
func WithDateFact(system string, now time.Time) string {
	if dateKeyRe.MatchString(system) || isoStampRe.MatchString(system) {
		return system
	}
	fact := now.Format("date: 2006-01-02 Mon 15:04")
	if strings.TrimSpace(system) == "" {
		return fact
	}
	if strings.HasPrefix(strings.TrimLeft(system, " \t\n"), "{") {
		return system
	}
	return fact + "; " + system
}

// walkGrounding checks every "date"/"date-time" value in arguments against the
// licensed years. It returns the paths it checked and the subset that failed,
// keyed argument-relative ("due_date", "items[0].when").
func walkGrounding(root map[string]any, arguments map[string]any, years map[int64]bool) (checked, failures map[string]bool) {
	checked = map[string]bool{}
	failures = map[string]bool{}
	if root == nil {
		return checked, failures
	}

	var walk func(value any, node map[string]any, path string)
	walk = func(value any, node map[string]any, path string) {
		node = soleVariant(resolveRef(node, root), root)

		if format, _ := node["format"].(string); (format == "date" || format == "date-time") && len(years) > 0 {
			if s, ok := value.(string); ok {
				if m := yearPrefixRe.FindStringSubmatch(s); m != nil {
					checked[path] = true
					if y, err := strconv.ParseInt(m[1], 10, 64); err == nil && !years[y] {
						failures[path] = true
					}
				}
			}
			return
		}

		switch v := value.(type) {
		case map[string]any:
			props, _ := node["properties"].(map[string]any)
			for key, item := range v {
				if sub, ok := props[key].(map[string]any); ok {
					walk(item, sub, joinPath(path, key))
				}
			}
		case []any:
			items, ok := node["items"].(map[string]any)
			if !ok {
				return
			}
			for i, item := range v {
				walk(item, items, path+"["+strconv.Itoa(i)+"]")
			}
		}
	}

	walk(arguments, root, "")
	return checked, failures
}

func joinPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// soleVariant collapses an anyOf/oneOf that only differs by a null branch, the
// shape a nullable pydantic field compiles to.
func soleVariant(node, root map[string]any) map[string]any {
	variants, _ := node["anyOf"].([]any)
	if variants == nil {
		variants, _ = node["oneOf"].([]any)
	}
	if len(variants) == 0 {
		return node
	}
	var concrete []map[string]any
	for _, raw := range variants {
		v, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if t, _ := resolveRef(v, root)["type"].(string); t != "null" {
			concrete = append(concrete, v)
		}
	}
	if len(concrete) == 1 {
		return resolveRef(concrete[0], root)
	}
	return node
}

// resolveRef follows a local "$ref" pointer ("#/$defs/LineItem") to its node.
func resolveRef(node, root map[string]any) map[string]any {
	seen := map[string]bool{}
	for node != nil {
		ref, ok := node["$ref"].(string)
		if !ok || seen[ref] {
			return node
		}
		seen[ref] = true
		target := root
		for _, part := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
			part = strings.ReplaceAll(part, "~1", "/")
			part = strings.ReplaceAll(part, "~0", "~")
			next, ok := target[part].(map[string]any)
			if !ok {
				return node
			}
			target = next
		}
		if target == nil {
			return node
		}
		node = target
	}
	return node
}

// dateFactRe matches the date facts injected into a system prompt, plus an
// ISO timestamp; they are context, not numbers in the record.
var dateFactRe = regexp.MustCompile(`date:\s*\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?`)

// numberLiteral is one numeric token found in a source text.
type numberLiteral struct {
	text string
}

// numberTokens scans a text for the numeric literals a reader would take away,
// the equivalent of the reference package's
//
//	(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)
//
// RE2 has no lookaround, so the guards are done by hand: a token may not start
// after a word character, "." or ",", and may not run into a digit. The
// thousands-grouped branch is tried first and rejected when its last group
// would be followed by a digit, which is what keeps "$1,2000.00" from minting
// 1200 (it mints 1 instead, exactly as the backtracking reference does).
func numberTokens(text string) []string {
	var out []string
	n := len(text)
	for i := 0; i < n; {
		c := text[i]
		signed := (c == '+' || c == '-') && i+1 < n && isDigit(text[i+1])
		if !isDigit(c) && !signed {
			i++
			continue
		}
		if i > 0 && isWordByte(text[i-1]) {
			i++
			continue
		}
		if tok, ok := matchGroupedNumber(text, i); ok {
			out = append(out, tok)
			i += len(tok)
			continue
		}
		if tok, ok := matchPlainNumber(text, i); ok {
			out = append(out, tok)
			i += len(tok)
			continue
		}
		i++
	}
	return out
}

// matchGroupedNumber matches [-+]\d{1,3}(,\d{3})+(\.\d+)? at start.
func matchGroupedNumber(text string, start int) (string, bool) {
	i := start
	if c := text[i]; c == '+' || c == '-' {
		i++
	}
	digits := 0
	for i < len(text) && isDigit(text[i]) && digits < 3 {
		i++
		digits++
	}
	if digits == 0 {
		return "", false
	}
	groups := 0
	for i < len(text) && text[i] == ',' {
		if i+4 > len(text) || !isDigit(text[i+1]) || !isDigit(text[i+2]) || !isDigit(text[i+3]) {
			break
		}
		i += 4
		groups++
		// A group must be complete and last: ",12000" is not thousands
		// grouping, so the grouped branch gives up and the plain one runs.
		if i < len(text) && isDigit(text[i]) {
			return "", false
		}
	}
	if groups == 0 {
		return "", false
	}
	if i < len(text) && text[i] == '.' {
		j := i + 1
		for j < len(text) && isDigit(text[j]) {
			j++
		}
		if j == i+1 {
			return "", false
		}
		i = j
	}
	if i < len(text) && isDigit(text[i]) {
		return "", false
	}
	return text[start:i], true
}

// matchPlainNumber matches [-+]\d+(\.\d+)? at start.
func matchPlainNumber(text string, start int) (string, bool) {
	i := start
	if c := text[i]; c == '+' || c == '-' {
		i++
	}
	digits := 0
	for i < len(text) && isDigit(text[i]) {
		i++
		digits++
	}
	if digits == 0 {
		return "", false
	}
	if i < len(text) && text[i] == '.' {
		j := i + 1
		for j < len(text) && isDigit(text[j]) {
			j++
		}
		if j == i+1 {
			return "", false
		}
		i = j
	}
	if i < len(text) && isDigit(text[i]) {
		return "", false
	}
	return text[start:i], true
}

// sourceNumbers collects the numeric values written in the source texts,
// thousands separators normalized and date facts stripped so a hyphen inside a
// date is not read as a minus sign.
func sourceNumbers(sources ...string) map[string]bool {
	cleaned := make([]string, 0, len(sources))
	for _, s := range sources {
		cleaned = append(cleaned, dateFactRe.ReplaceAllString(s, " "))
	}
	numbers := map[string]bool{}
	for _, s := range cleaned {
		if s == "" {
			continue
		}
		for _, tok := range numberTokens(s) {
			if key := normalizeNumber(tok); key != "" {
				numbers[key] = true
			}
		}
	}
	return numbers
}

// normalizeNumber renders a numeric literal (or a decoded JSON number) as an
// exact decimal key, so "1,200.00" and 1200 match while 12000 does not.
func normalizeNumber(raw string) string {
	raw = strings.ReplaceAll(raw, ",", "")
	r, ok := new(big.Rat).SetString(raw)
	if !ok {
		return ""
	}
	return r.RatString()
}

// numericLeaves calls out for every numeric leaf of value with its
// arguments-relative path and its normalized key. Booleans are skipped: they
// are answers, not quantities.
func numericLeaves(value any, path string, out func(path, key string)) {
	switch v := value.(type) {
	case bool:
	case float64:
		emitNumber(out, path, normalizeNumber(strconv.FormatFloat(v, 'g', -1, 64)))
	case int64:
		emitNumber(out, path, normalizeNumber(strconv.FormatInt(v, 10)))
	case json.Number:
		emitNumber(out, path, normalizeNumber(v.String()))
	case map[string]any:
		for key, item := range v {
			numericLeaves(item, joinPath(path, key), out)
		}
	case []any:
		for i, item := range v {
			numericLeaves(item, path+"["+strconv.Itoa(i)+"]", out)
		}
	}
}

func emitNumber(out func(path, key string), path, key string) {
	if path == "" || key == "" {
		return
	}
	out(path, key)
}

// groundedNumberPaths are the numeric argument paths whose value is written in
// one of the sources. With no numbers in the sources nothing is grounded, which
// keeps a fabricated number failing.
func groundedNumberPaths(arguments map[string]any, sources ...string) map[string]bool {
	numbers := sourceNumbers(sources...)
	if len(numbers) == 0 {
		return map[string]bool{}
	}
	grounded := map[string]bool{}
	numericLeaves(arguments, "", func(path, key string) {
		if numbers[key] {
			grounded[path] = true
		}
	})
	return grounded
}

// engineUngroundedPaths strips the tool prefix the engine qualifies its flagged
// names with, mirroring _ungrounded_paths.
func engineUngroundedPaths(v *Validation) (paths []string, negation bool) {
	if v == nil {
		return nil, false
	}
	for _, name := range v.Ungrounded {
		_, path, found := strings.Cut(name, ".")
		if !found {
			path = name
		}
		paths = append(paths, path)
	}
	return paths, v.Negation
}

// negationPath is the pseudo-path reported for a negated request.
const negationPath = "negated request"

// validateExtraction is the strict check: temporal grounding plus the engine's
// own flags, with numbers cleared only when they appear in the input.
func validateExtraction(text string, root map[string]any, arguments map[string]any, v *Validation, system string) error {
	checked, failures := walkGrounding(root, arguments,
		licensedYears(sourceYears(text), system, relativeCue(text)))

	flagged, negation := engineUngroundedPaths(v)
	var grounded map[string]bool
	if len(flagged) > 0 {
		grounded = groundedNumberPaths(arguments, text, system)
	}
	for _, path := range flagged {
		if grounded[path] {
			continue
		}
		// A flagged path the temporal check just cleared is the engine being
		// overruled by a year the input does license; anything else stands.
		if !checked[path] || failures[path] {
			failures[path] = true
		}
	}
	if negation {
		failures[negationPath] = true
	}
	if len(failures) == 0 {
		return nil
	}
	paths := make([]string, 0, len(failures))
	for p := range failures {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return &ExtractionError{Paths: paths}
}

// schemaParameters accepts either a bare parameter object or a tool entry
// wrapping one under "parameters", the two shapes a caller may hand us.
func schemaParameters(schema any) map[string]any {
	switch s := schema.(type) {
	case Schema:
		return s.Parameters
	case *Schema:
		if s == nil {
			return nil
		}
		return s.Parameters
	case map[string]any:
		if p, ok := s["parameters"].(map[string]any); ok {
			return p
		}
		return s
	case Tool:
		return s.Parameters
	}
	return nil
}

// ValidateGrounding is the strict check on its own, for callers who drive the
// conversation through Engine.Complete or Run instead of Extract: it reports
// whether arguments are grounded in text, given the schema and the environment
// facts the model saw. It returns an *ExtractionError, or nil.
//
// The engine's own report folds into the same check: pass resp.Validation
// (nil for the pure backend) to include its flags.
func ValidateGrounding(text string, schema any, arguments map[string]any, system string, engineReport *Validation) error {
	root := schemaParameters(schema)
	if root == nil {
		return errors.New("needle: grounding needs a schema with a parameter object")
	}
	return validateExtraction(text, root, arguments, engineReport, system)
}
