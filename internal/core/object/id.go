package object

import "sync/atomic"

var nextID uint64

func NewID() uint64 {
	return atomic.AddUint64(&nextID,1)
}