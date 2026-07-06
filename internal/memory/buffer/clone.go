package buffer

func (b *CPUBuffer) Clone() Buffer {

    cp := make([]byte, len(b.data))

    copy(cp, b.data)

    return &CPUBuffer{
        data:  cp,
        dtype: b.dtype,
    }
}