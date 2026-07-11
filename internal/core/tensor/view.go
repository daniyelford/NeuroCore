package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
)

func (t Tensor) View(
	newShape shape.Shape,
	offset int,
) (Tensor, bool) {

	if offset < 0 {

		return Tensor{}, false

	}

	if offset+newShape.NumElements() >
		t.memory.Len()-t.offset {

		return Tensor{}, false

	}

	return Tensor{

		shape: newShape,

		stride: stride.Compute(
			newShape,
			t.layout,
		),

		memory: t.memory,

		offset: t.offset + offset,

		device: t.device,

		layout: t.layout,
	}, true

}
func (t Tensor) Slice(
	start int,
	end int,
) (Tensor, bool) {

	dims :=
		t.shape.Values()

	if len(dims) == 0 {

		return Tensor{}, false

	}

	if start < 0 ||
		end > dims[0] ||
		start >= end {

		return Tensor{}, false

	}

	newDims :=
		make([]int, len(dims))

	copy(
		newDims,
		dims,
	)

	newDims[0] =
		end - start

	offset :=
		start * t.stride.At(0)

	return t.View(
		shape.New(newDims...),
		offset,
	)

}
