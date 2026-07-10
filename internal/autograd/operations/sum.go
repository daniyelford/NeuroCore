package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Sum struct {
	Base
}

func (op *Sum) Name() string {
	return "Sum"
}

func (op *Sum) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {
		return nil, errors.New("sum requires exactly one input")
	}

	x := inputs[0]

	op.Save(x)

	out := x.Data().SumTensor()
	v := autograd.NewVariable(
		out,
		x.RequiresGrad(),
	)

	op.SetOutput(v)

	return v, nil
}

func (op *Sum) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	// TODO: Broadcast گرادیان به شکل ورودی
	return nil, errors.New("sum backward not implemented")
}
