package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Sigmoid struct {
	Base
}

func (op *Sigmoid) Name() string {
	return "Sigmoid"
}

func (op *Sigmoid) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("sigmoid requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out := x.Data().Sigmoid()

	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)

	op.SetOutput(v)

	return v, nil
}

func (op *Sigmoid) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	output := op.Output().Data()

	out := output.SigmoidBackward(
		grad,
	)

	return []tensor.Tensor{
		out,
	}, nil
}
