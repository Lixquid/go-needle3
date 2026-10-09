//go:build windows && amd64

package cgo

// The windows engine archive is COFF objects built with clang targeting
// windows-gnu and linked against libc++; llvm-mingw provides matching
// static libc++/libc++abi archives. With plain mingw-w64 gcc, point -L at an
// llvm-mingw sysroot or link a libc++ for the target (see README).
//
// #cgo LDFLAGS: ${SRCDIR}/libs/windows-x86_64/libneedle.a -lc++ -lc++abi -lm
import "C"
