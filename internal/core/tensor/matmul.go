package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (t Tensor) MatMul(
	other Tensor,
) (Tensor, bool) {

	a := t.shape.Values()

	b := other.shape.Values()

	if len(a) != 2 || len(b) != 2 {

		return Tensor{}, false

	}

	if a[1] != b[0] {

		return Tensor{}, false

	}

	out := New(
		shape.New(
			a[0],
			b[1],
		),
	)

	for i := 0; i < a[0]; i++ {

		for j := 0; j < b[1]; j++ {

			var sum float32

			for k := 0; k < a[1]; k++ {

				av := t.At(i, k)

				bv := other.At(k, j)

				sum += av * bv

			}

			out.Set(
				sum,
				i,
				j,
			)

		}

	}

	return out, true

}
