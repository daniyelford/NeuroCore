package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
)

func (t Tensor) Reshape(
	newShape shape.Shape,
) (Tensor, bool) {

	if t.NumElements() != newShape.NumElements() {

		return Tensor{}, false

	}

	return Tensor{

		shape: newShape,

		stride: stride.Compute(
			newShape,
			t.layout,
		),

		memory: t.memory,
		offset: t.offset,
		device: t.device,

		layout: t.layout,
	}, true

}
