package buffer

import "github.com/daniyelford/neurocore/internal/core/dtype"

type Buffer interface{

	Pointer() unsafe.Pointer

	ByteSize() uintptr

	DType() dtype.DType

}
