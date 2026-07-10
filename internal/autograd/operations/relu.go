package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type ReLU struct {
	Base
}

func (op *ReLU) Name() string {
	return "ReLU"
}

func (op *ReLU) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("relu requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out := x.Data().ReLU()

	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)

	op.SetOutput(v)

	return v, nil
}

func (op *ReLU) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	input := op.Input(0).Data()

	mask := input.ReLUMask()

	out := grad.Mul(
		mask,
	)

	return []tensor.Tensor{
		out,
	}, nil
}
