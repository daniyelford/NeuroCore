package allocator

import (
    "github.com/daniyelford/neurocore/internal/core/dtype"
    "github.com/daniyelford/neurocore/internal/memory/buffer"
)

type Allocator interface {
    Allocate(
        elements int,
        dt dtype.DType,
    ) (buffer.Buffer, error)
}