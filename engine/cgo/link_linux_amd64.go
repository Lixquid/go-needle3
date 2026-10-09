//go:build linux && amd64

package cgo

// #cgo LDFLAGS: ${SRCDIR}/libs/linux-x86_64/libneedle.a -l:libc++.so.1 -l:libc++abi.so.1 -lm
import "C"
