package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
)

func (t Tensor) Shape() shape.Shape {

	return t.shape

}

func (t Tensor) Stride() stride.Stride {

	return t.stride

}

func (t Tensor) Device() backend.DeviceType {

	return t.device

}

func (t Tensor) Len() int {

	return t.shape.NumElements()

}
func (t Tensor) NumElements() int {

	return t.shape.NumElements()

}

func (t Tensor) Empty() bool {

	return t.memory.Empty()

}
func (t Tensor) Offset() int {

	return t.offset

}
