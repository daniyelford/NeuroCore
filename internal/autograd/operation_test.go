package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestAddOperation(
	t *testing.T,
) {

	a := NewVariable(
		tensor.New(
			shape.New(2),
		),
		true,
	)

	b := NewVariable(
		tensor.New(
			shape.New(2),
		),
		true,
	)

	if !a.RequiresGrad() ||
		!b.RequiresGrad() {

		t.Fatal()

	}

}
