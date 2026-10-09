// Package model embeds the needle3.cact weights so a binary can carry the
// whole model in memory with no runtime file dependencies.
package model

import (
	_ "embed"
)

// Needle3 holds the full needle3.cact model image (~35 MB).
//
//go:embed needle3.cact
var Needle3 []byte
