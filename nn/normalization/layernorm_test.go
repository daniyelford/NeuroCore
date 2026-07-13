package normalization

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestLayerNorm(t *testing.T) {

	x :=
		tensor.New(
			shape.New(4),
		)

	x.Set(1, 0)
	x.Set(2, 1)
	x.Set(3, 2)
	x.Set(4, 3)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	l :=
		New(4)

	out :=
		l.Forward(
			*v,
		)

	if out.Data().Len() != 4 {

		t.Fatal()

	}

}
