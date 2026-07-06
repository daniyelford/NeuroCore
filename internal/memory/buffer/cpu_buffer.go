package buffer

import "github.com/daniyelford/neurocore/internal/core/dtype"

type CPUBuffer struct {
    data  []byte
    dtype dtype.DType
}

func (b *CPUBuffer) Len() int {
    size := int(b.dtype.Size())

    if size == 0 {
        return 0
    }

    return len(b.data) / size
}

func (b *CPUBuffer) ByteLen() int {
    return len(b.data)
}

func (b *CPUBuffer) DType() dtype.DType {
    return b.dtype
}