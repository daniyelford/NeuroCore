package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestVariable(t *testing.T) {

	tensorData := tensor.New(
		shape.New(2),
	)

	v := NewVariable(
		tensorData,
		true,
	)

	if !v.RequiresGrad() {

		t.Fatal()

	}

}
func TestNoGrad(t *testing.T) {

	DisableGrad()

	if GradEnabled() {

		t.Fatal()

	}

	EnableGrad()

	if !GradEnabled() {

		t.Fatal()

	}

}
func TestDetach(t *testing.T) {

	v := NewVariable(
		tensor.New(
			shape.New(2),
		),
		true,
	)

	d := v.Detach()

	if d.RequiresGrad() {

		t.Fatal()

	}

}
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
