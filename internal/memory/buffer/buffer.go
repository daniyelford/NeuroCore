package buffer

import "github.com/daniyelford/neurocore/internal/core/dtype"

type Buffer interface {
	Len() int
	ByteLen() int
	DType() dtype.DType
	Clone() Buffer
}
