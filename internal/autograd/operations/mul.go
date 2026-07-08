package operations

import "github.com/daniyelford/neurocore/internal/autograd"

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
	grad autograd.Variable,
) []autograd.Variable {

	ga := grad.Data.Mul(
		op.B.Data,
	)

	gb := grad.Data.Mul(
		op.A.Data,
	)

	return []autograd.Variable{

		autograd.NewVariable(
			ga,
			false,
		),

		autograd.NewVariable(
			gb,
			false,
		),
	}

}
