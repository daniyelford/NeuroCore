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
