package pool

func Acquire(size int) []byte {

    buf := BytePool.Get().([]byte)

    if cap(buf) < size {

        return make([]byte, size)

    }

    return buf[:size]

}