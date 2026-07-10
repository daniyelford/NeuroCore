package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Flatten struct {
	Base
}

func (op *Flatten) Name() string {
	return "Flatten"
}

func (op *Flatten) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("flatten requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out, ok := x.Data().Flatten()
	if !ok {
		return nil, errors.New("flatten failed")
	}

	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)

	op.SetOutput(v)

	return v, nil
}

func (op *Flatten) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	in := op.Input(0)

	out, ok := grad.Reshape(
		in.Data().Shape(),
	)

	if !ok {
		return nil, errors.New("reshape failed")
	}

	return []tensor.Tensor{
		out,
	}, nil
}
