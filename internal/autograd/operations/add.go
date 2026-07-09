package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Add struct {
	Base
}

func (op Add) Name() string {

	return "add"

}

func (op *Add) Forward(
	a autograd.Variable,
	b autograd.Variable,
) autograd.Variable {

	op.Save(a, b)

	out := autograd.NewVariable(
		a.Data.Add(b.Data),
		a.RequiresGrad || b.RequiresGrad,
	)

	op.SetOutput(out)

	return out

}
func (op *Add) Backward(
	grad tensor.Tensor,
) []tensor.Tensor {

	return []tensor.Tensor{
		grad,
		grad,
	}

}
