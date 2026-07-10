package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestLinear(
	t *testing.T,
) {

	layer :=
		NewLinear(
			3,
			2,
		)

	x :=
		*autograd.NewVariable(
			tensor.New(
				shape.New(1, 3),
			),
			false,
		)

	y :=
		layer.Forward(x)

	if y.Data().Shape().Values()[1] != 2 {

		t.Fatal()

	}

}
