package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Neg struct {
	Base
}

func (op *Neg) Name() string {
	return "Neg"
}

func (op *Neg) Forward(inputs ...*autograd.Variable) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("neg requires exactly 1 input")
	}

	x := inputs[0]

	op.Save(x)

	out := autograd.NewVariable(
		x.Data().Neg(),
		x.RequiresGrad(),
	)
	out.Node().Parents = []*autograd.Node{
		x.Node(),
	}

	out.Node().Op = op
	op.SetOutput(out)

	return out, nil
}

func (op *Neg) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	return []tensor.Tensor{
		grad.Neg(),
	}, nil
}
