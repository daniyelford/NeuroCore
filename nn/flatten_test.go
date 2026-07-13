package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestFlatten(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				2,
				3,
				4,
			),
		)

	f :=
		NewFlatten()

	out :=
		f.Forward(
			*autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().Shape().Values()

	if s[0] != 2 || s[1] != 12 {

		t.Fatalf(
			"wrong shape %v",
			s,
		)

	}

}
