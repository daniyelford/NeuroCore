package init

import (
	"math"

	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Xavier struct{}

func (x Xavier) Init(
	t *tensor.Tensor,
) {

	shape :=
		t.Shape().Values()

	if len(shape) != 2 {
		panic("xavier requires 2D tensor")
	}

	fanIn :=
		float32(shape[0])

	fanOut :=
		float32(shape[1])

	limit :=
		float32(
			math.Sqrt(
				float64(
					6.0 /
						(fanIn + fanOut),
				),
			),
		)

	Uniform{
		Min: -limit,
		Max: limit,
	}.Init(t)

}
