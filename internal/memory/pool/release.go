package pool

func Release(buf []byte) {

    BytePool.Put(buf[:0])

}