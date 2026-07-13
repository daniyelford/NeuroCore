package nn

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestConv2DStride(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				6,
				6,
			),
		)

	conv :=
		NewConv2D(
			1,
			1,
			3,
			3,
			2,
			0,
		)

	out :=
		conv.Forward(
			autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[2] != 2 ||
		s[3] != 2 {

		t.Fatal(
			"wrong stride output shape",
		)

	}

}
func TestConv2DSamePadding(t *testing.T) {

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				5,
				5,
			),
		)

	conv :=
		NewConv2D(
			1,
			1,
			3,
			3,
			1,
			1,
		)

	out :=
		conv.Forward(
			autograd.NewVariable(
				x,
				false,
			),
		)

	s :=
		out.Data().
			Shape().
			Values()

	if s[2] != 5 ||
		s[3] != 5 {

		t.Fatal(
			"same padding failed",
		)

	}

}
func TestConv2DGraph(t *testing.T) {

	conv :=
		NewConv2D(
			1,
			1,
			3,
			3,
			1,
			1,
		)

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				5,
				5,
			),
		)

	input :=
		autograd.NewVariable(
			x,
			true,
		)

	out :=
		conv.Forward(
			input,
		)

	if out.Node().Op == nil {

		t.Fatal(
			"conv not connected to graph",
		)

	}

	if len(out.Node().Parents) != 3 {

		t.Fatal(
			"conv parents missing",
		)

	}

}

// func TestConv2DBackward(t *testing.T) {

// 	conv :=
// 		NewConv2D(
// 			1,
// 			1,
// 			2,
// 			2,
// 			1,
// 			0,
// 		)

// 	x :=
// 		tensor.New(
// 			shape.New(
// 				1,
// 				1,
// 				3,
// 				3,
// 			),
// 		)

// 	input :=
// 		autograd.NewVariable(
// 			x,
// 			true,
// 		)

// 	out :=
// 		conv.Forward(
// 			input,
// 		)

// 	loss :=
// 		operations.NewSum()

// 	y, err :=
// 		loss.Forward(
// 			out,
// 		)

// 	if err != nil {
// 		t.Fatal(err)
// 	}

// 	autograd.Backward(
// 		y,
// 	)

// 	if input.Grad().Len() == 0 {

// 		t.Fatal(
// 			"input grad empty",
// 		)

// 	}

// 	if conv.Weight.Value.Grad().Len() == 0 {

// 		t.Fatal(
// 			"weight grad empty",
// 		)

// 	}

// 	if conv.Bias.Value.Grad().Len() == 0 {

// 		t.Fatal(
// 			"bias grad empty",
// 		)

// 	}

// }
