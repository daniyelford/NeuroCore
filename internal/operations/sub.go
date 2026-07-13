package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Sub struct {
	Base
}

func (op *Sub) Name() string {
	return "Sub"
}

func (op *Sub) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {

	if len(inputs) != 2 {
		return nil, errors.New("sub requires exactly 2 inputs")
	}

	a := inputs[0]
	b := inputs[1]

	op.Save(a, b)

	out := autograd.NewVariable(
		a.Data().Sub(b.Data()),
		a.RequiresGrad() || b.RequiresGrad(),
	)
	out.Node().Parents = []*autograd.Node{
		a.Node(),
		b.Node(),
	}

	out.Node().Op = op
	op.SetOutput(out)

	return out, nil
}

func (op *Sub) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	return []tensor.Tensor{
		grad,
		grad.Neg(),
	}, nil
}
