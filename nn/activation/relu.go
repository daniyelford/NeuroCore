package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type ReLU struct{}

func NewReLU() ReLU {

	return ReLU{}

}

func (r ReLU) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	data := input.Data()

	out := tensor.New(
		data.Shape(),
	)

	for i := 0; i < data.Len(); i++ {

		v := data.At(i)

		if v > 0 {

			out.Set(
				v,
				i,
			)

		} else {

			out.Set(
				0,
				i,
			)

		}

	}

	return autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}
