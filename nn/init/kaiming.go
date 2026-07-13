package init

import (
	"math"

	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Kaiming struct{}

func (k Kaiming) Init(
	t *tensor.Tensor,
) {

	shape :=
		t.Shape().Values()

	if len(shape) != 2 {
		panic("kaiming requires 2D tensor")
	}

	fanIn :=
		float32(shape[0])

	std :=
		float32(
			math.Sqrt(
				2.0 /
					float64(fanIn),
			),
		)

	Normal{
		Mean: 0,
		Std:  std,
	}.Init(t)

}
