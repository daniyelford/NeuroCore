package operations

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestConv2DBackward(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				4,
				4,
			),
		)

	w :=
		tensor.New(
			shape.New(
				1,
				1,
				3,
				3,
			),
		)

	b :=
		tensor.New(
			shape.New(
				1,
			),
		)

	input :=
		autograd.NewVariable(
			x,
			true,
		)

	weight :=
		autograd.NewVariable(
			w,
			true,
		)

	bias :=
		autograd.NewVariable(
			b,
			true,
		)

	op :=
		NewConv2D(
			1,
			1,
			0,
			0,
			3,
			3,
		)

	out, err :=
		op.Forward(
			input,
			weight,
			bias,
		)

	if err != nil {

		t.Fatal(err)

	}

	lossOp :=
		&Sum{}

	loss, err :=
		lossOp.Forward(
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

	if weight.Grad().Len() == 0 {

		t.Fatal(
			"weight grad empty",
		)

	}

	if bias.Grad().Len() == 0 {

		t.Fatal(
			"bias grad empty",
		)

	}

}
