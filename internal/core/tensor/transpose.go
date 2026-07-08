package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (t Tensor) Transpose() (Tensor, bool) {

	d := t.shape.Values()

	if len(d) != 2 {

		return Tensor{}, false

	}

	out := New(
		shape.New(
			d[1],
			d[0],
		),
	)

	for i := 0; i < d[0]; i++ {

		for j := 0; j < d[1]; j++ {

			out.Set(
				t.At(i, j),
				j,
				i,
			)

		}

	}

	return out, true

}
