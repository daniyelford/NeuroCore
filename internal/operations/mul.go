package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Mul struct {
	Base
}

func (op *Mul) Name() string {
	return "Mul"
}

func (op *Mul) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {

	if len(inputs) != 2 {
		return nil, errors.New("mul requires exactly 2 inputs")
	}

	a := inputs[0]
	b := inputs[1]

	op.Save(a, b)

	out := autograd.NewVariable(
		a.Data().Mul(b.Data()),
		a.RequiresGrad() || b.RequiresGrad(),
	)

	op.SetOutput(out)

	return out, nil
}

func (op *Mul) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	return []tensor.Tensor{
		grad.Mul(op.Input(1).Data()),
		grad.Mul(op.Input(0).Data()),
	}, nil
}
