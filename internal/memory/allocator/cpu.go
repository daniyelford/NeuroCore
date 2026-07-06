package allocator

import (
    "github.com/daniyelford/neurocore/internal/core/dtype"
    "github.com/daniyelford/neurocore/internal/memory/buffer"
)

type CPUAllocator struct{}

func (CPUAllocator) Allocate(
    elements int,
    dt dtype.DType,
) (buffer.Buffer, error) {

    return buffer.NewCPU(
        elements,
        dt,
    )
}