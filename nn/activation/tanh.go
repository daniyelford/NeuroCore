package activation

import (
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Tanh struct{}

func NewTanh() Tanh {

	return Tanh{}

}

func (t Tanh) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	out := tensor.New(
		input.Data().Shape(),
	)

	for i := 0; i < input.Data().Len(); i++ {

		v := math.Tanh(
			float64(
				input.Data().At(i),
			),
		)

		out.Set(
			float32(v),
			i,
		)

	}

	return autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}
