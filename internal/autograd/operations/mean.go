package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Mean struct {
	Base
}

func (op *Mean) Name() string {
	return "Mean"
}

func (op *Mean) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New(
			"mean requires exactly one input",
		)
	}

	x := inputs[0]

	op.Save(x)

	out := x.Data().ReduceMean()

	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)

	op.SetOutput(v)

	return v, nil
}

func (op *Mean) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	input := op.Input(0).Data()

	size := float32(
		input.NumElements(),
	)

	g := grad.DivScalar(
		size,
	)

	out, ok := g.Broadcast(
		input.Shape(),
	)

	if !ok {
		return nil, errors.New(
			"mean backward broadcast failed",
		)
	}

	return []tensor.Tensor{
		out,
	}, nil
}
