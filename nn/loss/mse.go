package loss

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type MSE struct{}

func NewMSE() MSE {

	return MSE{}

}

func (m MSE) Forward(
	prediction autograd.Variable,
	target autograd.Variable,
) autograd.Variable {

	diff :=
		prediction.Data.Sub(
			target.Data,
		)

	sum := float32(0)

	for i := 0; i < diff.Len(); i++ {

		v := diff.At(i)

		sum += v * v

	}

	result :=
		sum /
			float32(diff.Len())

	out := tensor.New(
		shape.New(1),
	)

	out.Set(
		result,
		0,
	)

	return autograd.NewVariable(
		out,
		true,
	)

}
