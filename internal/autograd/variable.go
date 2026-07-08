package autograd

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Variable struct {
	Data tensor.Tensor

	Grad tensor.Tensor

	RequiresGrad bool
}

func NewVariable(
	t tensor.Tensor,
	requires bool,
) Variable {

	return Variable{

		Data: t,

		RequiresGrad: requires,
	}

}
