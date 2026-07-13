package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Reshape struct {
	Base

	newShape shape.Shape
}

func NewReshape(
	s shape.Shape,
) *Reshape {

	return &Reshape{
		newShape: s,
	}
}

func (op *Reshape) Name() string {
	return "Reshape"
}

func (op *Reshape) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("reshape requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out, ok := x.Data().Reshape(
		op.newShape,
	)

	if !ok {
		return nil, errors.New(
			"invalid reshape",
		)
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

func (op *Reshape) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	original := op.Input(0).Data().Shape()

	out, ok := grad.Reshape(
		original,
	)

	if !ok {
		return nil, errors.New(
			"reshape backward failed",
		)
	}

	return []tensor.Tensor{
		out,
	}, nil
}
