package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type MatMul struct {
	A autograd.Variable
	B autograd.Variable
}

func (op MatMul) Backward(
	grad tensor.Tensor,
) []tensor.Tensor {

	bt, _ := op.B.Data().Transpose()

	at, _ := op.A.Data().Transpose()

	ga, _ := grad.MatMul(bt)

	gb, _ := at.MatMul(grad)

	return []tensor.Tensor{
		ga,
		gb,
	}
}
