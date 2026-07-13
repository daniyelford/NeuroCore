package conv

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func TestConv2DForward(t *testing.T) {

	conv :=
		NewConv2D(
			1,
			1,
			1,
			1,
		)

	// weight = 1
	conv.Weight.Value.Data().Set(
		1,
		0,
		0,
		0,
		0,
	)

	// bias = 0
	conv.Bias.Value.Data().Set(
		0,
		0,
	)

	x :=
		tensor.New(
			shape.New(
				1,
				1,
				3,
				3,
			),
		)

	value := float32(1)

	for i := 0; i < 3; i++ {

		for j := 0; j < 3; j++ {

			x.Set(
				value,
				0,
				0,
				i,
				j,
			)

			value++

		}

	}

	input :=
		autograd.NewVariable(
			x,
			false,
		)

	out :=
		conv.Forward(
			*input,
		)

	expected :=
		[]float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}

	index := 0

	for i := 0; i < 3; i++ {

		for j := 0; j < 3; j++ {

			got :=
				out.Data().At(
					0,
					0,
					i,
					j,
				)

			if got != expected[index] {

				t.Fatalf(
					"wrong output at %d got %v expected %v",
					index,
					got,
					expected[index],
				)

			}

			index++

		}

	}

}
