package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Add struct {
	A autograd.Variable

	B autograd.Variable
}

func NewAdd(
	a autograd.Variable,
	b autograd.Variable,
) Add {

	return Add{

		A: a,

		B: b,
	}

}

func (op Add) Name() string {

	return "add"

}

func (op Add) Forward() autograd.Variable {

	result := op.A.Data.Add(
		op.B.Data,
	)

	return autograd.NewVariable(
		result,
		op.A.RequiresGrad ||
			op.B.RequiresGrad,
	)

}

func (op Add) Backward(
	grad tensor.Tensor,
) []tensor.Tensor {

	return []tensor.Tensor{

		grad,

		grad,
	}

}
