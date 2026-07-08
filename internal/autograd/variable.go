package autograd

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Variable struct {
	Data tensor.Tensor

	Grad tensor.Tensor

	RequiresGrad bool

	node *Node
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
func (v Variable) Node() *Node {

	if v.node != nil {

		return v.node

	}

	return &Node{

		Output: v,
	}

}
