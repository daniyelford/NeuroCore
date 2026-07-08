package device

import (
	"github.com/daniyelford/neurocore/internal/core/memory"
)

type Device interface {
	Type() Type

	Name() string

	Allocate(size int) memory.Memory

	Copy(
		dst memory.Memory,
		src memory.Memory,
	)
}
