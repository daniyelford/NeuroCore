package autograd

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

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
