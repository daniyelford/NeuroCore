package operations

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestBatchNormBackward(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				2,
				3,
				2,
				2,
			),
		)

	gamma :=
		tensor.New(
			shape.New(
				3,
			),
		)

	beta :=
		tensor.New(
			shape.New(
				3,
			),
		)

	input :=
		autograd.NewVariable(
			x,
			true,
		)

	g :=
		autograd.NewVariable(
			gamma,
			true,
		)

	b :=
		autograd.NewVariable(
			beta,
			true,
		)

	op :=
		NewBatchNorm(
			3,
			1e-5,
		)

	out, err :=
		op.Forward(
			input,
			g,
			b,
		)

	if err != nil {
		t.Fatal(err)
	}

	sum :=
		&Sum{}

	loss, err :=
		sum.Forward(
			out,
		)

	if err != nil {
		t.Fatal(err)
	}

	autograd.Backward(
		loss,
	)

	if input.Grad().Len() == 0 {

		t.Fatal(
			"input grad empty",
		)

	}

	if g.Grad().Len() == 0 {

		t.Fatal(
			"gamma grad empty",
		)

	}

	if b.Grad().Len() == 0 {

		t.Fatal(
			"beta grad empty",
		)

	}

}
