package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestModelParameters(
	t *testing.T,
) {

	layer :=
		NewLinear(
			3,
			2,
		)

	model :=
		NewModel(
			layer,
		)

	if len(model.Parameters()) != 2 {

		t.Fatal()

	}

}
func TestSequential(t *testing.T) {

	model := NewSequential(
		NewLinear(3, 4),
		NewLinear(4, 2),
	)

	x := autograd.NewVariable(
		tensor.New(shape.New(1, 3)),
		false,
	)

	y := model.Forward(*x)

	if y.Data().Shape().Values()[1] != 2 {
		t.Fatal()
	}

}
