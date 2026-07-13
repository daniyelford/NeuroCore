package pool

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestMaxPool2D(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				2,
				2,
			),
		)

	x.Set(1, 0, 0, 0, 0)
	x.Set(2, 0, 0, 0, 1)
	x.Set(3, 0, 0, 1, 0)
	x.Set(4, 0, 0, 1, 1)

	pool :=
		NewMaxPool2D(
			2,
			2,
		)

	out :=
		pool.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	if out.Data().FlatAt(0) != 4 {

		t.Fatalf(
			"expected 4 got %f",
			out.Data().FlatAt(0),
		)

	}

}
