package dropout

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestDropoutTrain(t *testing.T) {

	x :=
		tensor.New(
			shape.New(10),
		)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	d :=
		New(
			0.5,
		)

	out :=
		d.Forward(
			*v,
		)

	if out.Data().Len() != 10 {

		t.Fatal()

	}

}

func TestDropoutEval(t *testing.T) {

	x :=
		tensor.New(
			shape.New(5),
		)

	v :=
		autograd.NewVariable(
			x,
			false,
		)

	d :=
		New(
			0.5,
		)

	d.Eval()

	out :=
		d.Forward(
			*v,
		)

	if !out.Data().Shape().Equal(
		x.Shape(),
	) {

		t.Fatal()

	}

}
