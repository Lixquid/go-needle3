//go:build linux && amd64

package cgo

// Static linking: libc++/libc++abi are linked statically so the final binary
// has no shared-library requirement beyond the base system (libc/libm/libgcc_s).

// #cgo LDFLAGS: ${SRCDIR}/libs/linux-x86_64/libneedle.a -l:libc++.a -l:libc++abi.a -lm
import "C"
