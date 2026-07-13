package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Transpose struct {
	Base
}

func (op *Transpose) Name() string {
	return "Transpose"
}

func (op *Transpose) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("transpose requires exactly 1 input")
	}

	x := inputs[0]

	op.Save(x)

	out, ok := x.Data().Transpose()
	if !ok {
		return nil, errors.New("cannot transpose tensor")
	}

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

func (op *Transpose) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	out, ok := grad.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose gradient")
	}

	return []tensor.Tensor{
		out,
	}, nil
}
