package activation

import (
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Sigmoid struct{}

func NewSigmoid() Sigmoid {

	return Sigmoid{}

}

func (s Sigmoid) Forward(
	input autograd.Variable,
) autograd.Variable {

	out := tensor.New(
		input.Data.Shape(),
	)

	for i := 0; i < input.Data.Len(); i++ {

		x := input.Data.At(i)

		y := float32(
			1.0 /
				(1.0 + math.Exp(float64(-x))),
		)

		out.Set(
			y,
			i,
		)

	}

	return autograd.NewVariable(
		out,
		input.RequiresGrad,
	)

}
