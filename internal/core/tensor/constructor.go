package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
	"github.com/daniyelford/neurocore/internal/memory"
)

func New(
	sh shape.Shape,
) Tensor {

	size := sh.NumElements()

	return Tensor{

		shape: sh,

		stride: stride.Compute(
			sh,
			layout.RowMajor,
		),

		memory: memory.New(size),

		device: backend.CPU,

		layout: layout.RowMajor,
	}

}
func From(
	sh shape.Shape,
	values []float32,
) Tensor {

	t := New(sh)

	t.memory.CopyFrom(values)

	return t

}
