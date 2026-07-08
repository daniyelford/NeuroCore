package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Mul struct {
	A autograd.Variable

	B autograd.Variable
}

func NewMul(
	a autograd.Variable,
	b autograd.Variable,
) Mul {

	return Mul{

		A: a,

		B: b,
	}

}

func (op Mul) Name() string {

	return "mul"

}

func (op Mul) Forward() autograd.Variable {

	result := op.A.Data.Mul(
		op.B.Data,
	)

	return autograd.NewVariable(
		result,
		true,
	)

}

func (op Mul) Backward(
	grad tensor.Tensor,
) []tensor.Tensor {

	ga :=
		grad.Mul(
			op.B.Data,
		)

	gb :=
		grad.Mul(
			op.A.Data,
		)

	return []tensor.Tensor{

		ga,

		gb,
	}

}
