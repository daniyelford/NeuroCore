package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestAddGraph(
	t *testing.T,
) {

	a :=
		NewVariable(
			tensor.New(
				shape.New(2),
			),
			true,
		)

	b :=
		NewVariable(
			tensor.New(
				shape.New(2),
			),
			true,
		)

	c :=
		Add(
			a,
			b,
		)

	if !c.RequiresGrad() {

		t.Fatal()

	}

	Backward(c)

	if c.Grad().Empty() {

		t.Fatal()

	}

}
