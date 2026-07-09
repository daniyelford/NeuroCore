package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

func Accumulate(
	node *Node,
	grad tensor.Tensor,
) {
	if node.Grad.Empty() {
		node.Grad = grad
		return
	}

	node.Grad = node.Grad.Add(grad)
}
