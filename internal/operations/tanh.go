package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Tanh struct {
	Base
}

func (op *Tanh) Name() string {
	return "Tanh"
}

func (op *Tanh) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("tanh requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out := x.Data().Tanh()

	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)
	v.Node().Parents = []*autograd.Node{
		x.Node(),
	}

	v.Node().Op = op
	op.SetOutput(v)

	return v, nil
}

func (op *Tanh) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	output := op.Output().Data()

	out := output.TanhBackward(
		grad,
	)

	return []tensor.Tensor{
		out,
	}, nil
}
