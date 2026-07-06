package pool

import "sync"

var BytePool = sync.Pool{

    New: func() any {

        return make([]byte, 0)

    },
}