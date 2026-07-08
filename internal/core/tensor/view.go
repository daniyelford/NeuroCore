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

	if start < 0 ||
		end > t.NumElements() ||
		start >= end {

		return Tensor{}, false

	}

	return t.View(
		shape.New(end-start),
		start,
	)

}
