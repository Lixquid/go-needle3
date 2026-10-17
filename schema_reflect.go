package needle

// Deriving a JSON schema from a Go type, the stand-in for the pydantic model
// the Python package takes as its extraction schema.

import (
	"errors"
	"reflect"
	"strconv"
	"strings"
	"time"
)

var timeType = reflect.TypeOf(time.Time{})

// schemaFromType derives a Schema from a struct, map or pointer type.
func schemaFromType(t reflect.Type) (Schema, error) {
	if t == nil {
		return Schema{}, errors.New("needle: extract schema type is nil")
	}
	return schemaForType(t).normalized(), nil
}

// schemaForType is SchemaOf's shared body.
func schemaForType(t reflect.Type) Schema {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	sc := Schema{Name: typeName(t), Parameters: jsonSchemaFor(t, map[reflect.Type]bool{})}
	if sc.Name == "" {
		sc.Name = DefaultSchemaName
	}
	if desc, ok := describeType(t); ok {
		sc.Description = desc
	}
	if kind, _ := sc.Parameters["type"].(string); kind != "object" {
		// A scalar is not a record: the engine wants an object, so hand it one
		// and let Decode sort out the fields.
		sc.Parameters = map[string]any{"type": "object"}
	}
	return sc.normalized()
}

func typeName(t reflect.Type) string {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Name()
}

// describeType calls Description() on a zero value when the type offers one.
func describeType(t reflect.Type) (string, bool) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	pt := reflect.PointerTo(t)
	switch {
	case pt.Implements(describedType):
		if d, ok := reflect.New(t).Interface().(Described); ok {
			return d.Description(), true
		}
	case t.Implements(describedType):
		if d, ok := reflect.New(t).Elem().Interface().(Described); ok {
			return d.Description(), true
		}
	}
	return "", false
}

var describedType = reflect.TypeOf((*Described)(nil)).Elem()

// jsonSchemaFor maps one Go type onto a JSON-schema fragment. seen guards
// against recursive types, which the engine's grammar cannot express anyway.
func jsonSchemaFor(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if seen[t] {
		return map[string]any{"type": "object"}
	}

	// Types that describe themselves.
	if s, ok := providedSchema(t); ok {
		return s
	}
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 && t.Kind() == reflect.Slice {
			return map[string]any{"type": "string"} // []byte travels as a JSON string
		}
		return map[string]any{"type": "array", "items": jsonSchemaFor(t.Elem(), seen)}
	case reflect.Map:
		return map[string]any{"type": "object"}
	case reflect.Struct:
		seen[t] = true
		defer delete(seen, t)
		return objectSchemaFor(t, seen)
	default:
		// Interfaces and channels: the engine's fallback is a free-text string.
		return map[string]any{"type": "string"}
	}
}

// providedSchema uses SchemaProvider when the type (or its pointer) implements
// it, calling the method on a zero value.
func providedSchema(t reflect.Type) (map[string]any, bool) {
	pt := reflect.PointerTo(t)
	switch {
	case pt.Implements(schemaProviderType):
		if p, ok := reflect.New(t).Interface().(SchemaProvider); ok {
			return p.JSONSchema(), true
		}
	case t.Implements(schemaProviderType):
		if p, ok := reflect.New(t).Elem().Interface().(SchemaProvider); ok {
			return p.JSONSchema(), true
		}
	}
	return nil, false
}

var schemaProviderType = reflect.TypeOf((*SchemaProvider)(nil)).Elem()

// objectSchemaFor builds {"type":"object","properties":…,"required":…} for a
// struct, following encoding/json's name rules and the extraction tags.
func objectSchemaFor(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	props := map[string]any{}
	var required []string
	for _, f := range structFields(t) {
		field := jsonSchemaFor(f.Type, seen)
		applyFieldTags(field, f.Tag, f.optional)
		props[f.Name] = field
		if f.required {
			required = append(required, f.Name)
		}
	}
	out := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		out["required"] = required
	}
	return out
}

type structField struct {
	Name     string
	Type     reflect.Type
	Tag      reflect.StructTag
	required bool
	optional bool
}

// structFields resolves the exported fields of a struct the way
// encoding/json does: json tags name them, "-" drops them, and unnamed
// embedded structs are inlined.
func structFields(t reflect.Type) []structField {
	var out []structField
	var walk func(t reflect.Type, depth int)
	walk = func(t reflect.Type, depth int) {
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			tag := f.Tag.Get("json")
			name := strings.Split(tag, ",")[0]
			if name == "-" {
				continue
			}
			if f.Anonymous && name == "" {
				ft := f.Type
				for ft.Kind() == reflect.Pointer {
					ft = ft.Elem()
				}
				if ft.Kind() == reflect.Struct && ft != timeType {
					walk(ft, depth+1)
					continue
				}
			}
			if !f.IsExported() {
				continue
			}
			if name == "" {
				name = f.Name
			}
			sf := structField{Name: name, Type: f.Type, Tag: f.Tag}
			// A pointer, an omitempty/omitzero tag or a declared default makes
			// the field optional, mirroring the Python package's rule that a
			// field with a default is not required.
			_, hasDefault := f.Tag.Lookup("default")
			sf.optional = f.Type.Kind() == reflect.Pointer || hasDefault ||
				strings.Contains(tag, ",omitempty") || strings.Contains(tag, ",omitzero")
			if req, ok := f.Tag.Lookup("required"); ok {
				sf.required = req == "true"
				sf.optional = req == "false"
			} else {
				sf.required = !sf.optional
			}
			out = append(out, sf)
		}
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return out
	}
	walk(t, 0)
	return out
}

// applyFieldTags folds the extraction-specific tags into a property schema:
//
//	description "…"  what the field means, shown to the model
//	format      "…"  "date" or "date-time" turns on temporal grounding
//	enum        "…"  comma-separated closed value set (classification)
//	default     "…"  implied value; also makes the field optional
//	minimum/-imum    numeric bounds, "maximum" likewise
func applyFieldTags(schema map[string]any, tag reflect.StructTag, optional bool) {
	if d, ok := tag.Lookup("description"); ok && d != "" {
		schema["description"] = d
	}
	if f, ok := tag.Lookup("format"); ok && f != "" {
		schema["format"] = f
	}
	if e, ok := tag.Lookup("enum"); ok && e != "" {
		values := strings.Split(e, ",")
		for i := range values {
			values[i] = strings.TrimSpace(values[i])
		}
		schema["enum"] = values
		if t, _ := schema["type"].(string); t == "integer" || t == "number" {
			typed := make([]any, 0, len(values))
			for _, v := range values {
				if n, err := strconv.ParseFloat(v, 64); err == nil {
					if t == "integer" {
						typed = append(typed, int64(n))
					} else {
						typed = append(typed, n)
					}
				} else {
					typed = append(typed, v)
				}
			}
			schema["enum"] = typed
		}
	}
	for key, tagKey := range map[string]string{"minimum": "minimum", "maximum": "maximum"} {
		raw, ok := tag.Lookup(tagKey)
		if !ok {
			continue
		}
		if t, _ := schema["type"].(string); t == "integer" {
			if n, err := strconv.ParseInt(raw, 10, 64); err == nil {
				schema[key] = n
				continue
			}
		}
		if n, err := strconv.ParseFloat(raw, 64); err == nil {
			schema[key] = n
		}
	}
	if d, ok := tag.Lookup("default"); ok {
		schema["default"] = coerceDefault(d, schema["type"])
	}
}

func coerceDefault(d string, typ any) any {
	switch typ {
	case "integer":
		if n, err := strconv.ParseInt(d, 10, 64); err == nil {
			return n
		}
	case "number":
		if n, err := strconv.ParseFloat(d, 64); err == nil {
			return n
		}
	case "boolean":
		if b, err := strconv.ParseBool(d); err == nil {
			return b
		}
	}
	return d
}
