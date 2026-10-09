//go:build windows && arm64

package cgo

// #cgo LDFLAGS: ${SRCDIR}/libs/windows-arm64/libneedle.a -lc++ -lc++abi -lm
import "C"
