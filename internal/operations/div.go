package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Div struct {
	Base
}

func (op *Div) Name() string {
	return "Div"
}

func (op *Div) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {

	if len(inputs) != 2 {
		return nil, errors.New("div requires exactly 2 inputs")
	}

	a := inputs[0]
	b := inputs[1]

	op.Save(a, b)

	out := autograd.NewVariable(
		a.Data().Div(b.Data()),
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

func (op *Div) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	a := op.Input(0).Data()
	b := op.Input(1).Data()

	return []tensor.Tensor{
		grad.Div(b),
		grad.Mul(a).Div(b.Mul(b)).Neg(),
	}, nil
}
