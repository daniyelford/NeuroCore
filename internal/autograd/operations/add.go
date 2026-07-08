package operations

import (
	"github.com/daniyelford/neurocore/internal/core/autograd"
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
	grad autograd.Variable,
) []autograd.Variable {

	return []autograd.Variable{

		autograd.NewVariable(
			grad.Data,
			false,
		),

		autograd.NewVariable(
			grad.Data,
			false,
		),
	}

}
