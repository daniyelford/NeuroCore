package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type MatMul struct {
	Base
}

func (op *MatMul) Name() string {
	return "MatMul"
}

func (op *MatMul) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 2 {
		return nil, errors.New("matmul requires exactly 2 inputs")
	}

	a := inputs[0]
	b := inputs[1]

	op.Save(a, b)

	out, ok := a.Data().MatMul(
		b.Data(),
	)

	if !ok {
		return nil, errors.New("invalid matrix multiplication")
	}

	v := autograd.NewVariable(
		out,
		a.RequiresGrad() || b.RequiresGrad(),
	)
	v.Node().Parents = []*autograd.Node{
		a.Node(),
		b.Node(),
	}

	v.Node().Op = op
	op.SetOutput(v)

	return v, nil
}

func (op *MatMul) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	a := op.Input(0).Data()
	b := op.Input(1).Data()

	bt, ok := b.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose rhs")
	}

	at, ok := a.Transpose()
	if !ok {
		return nil, errors.New("cannot transpose lhs")
	}

	da, ok := grad.MatMul(bt)
	if !ok {
		return nil, errors.New("cannot compute lhs gradient")
	}

	db, ok := at.MatMul(grad)
	if !ok {
		return nil, errors.New("cannot compute rhs gradient")
	}

	return []tensor.Tensor{
		da,
		db,
	}, nil
}
