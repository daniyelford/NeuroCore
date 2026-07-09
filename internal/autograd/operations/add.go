package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Add struct {
	Base
}

func (op *Add) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	a := inputs[0]
	b := inputs[1]

	op.SaveInputs(a, b)

	outData := a.Data().Add(
		b.Data(),
	)

	out := autograd.NewVariable(
		outData,
		a.RequiresGrad() || b.RequiresGrad(),
	)

	op.SetOutput(out)

	return out, nil
}
func (op *Add) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	return []tensor.Tensor{
		grad,
		grad,
	}, nil

}
