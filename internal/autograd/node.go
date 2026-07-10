package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Node struct {

	// مقدار Tensor
	Data tensor.Tensor

	// گرادیان
	Grad tensor.Tensor

	// آیا نیاز به گرادیان دارد؟
	RequiresGrad bool

	// نودهای والد
	Parents []*Node

	// عملیاتی که این نود را تولید کرده است
	Op Operation

	// شناسه اختیاری برای Debug
	ID uint64
}

func NewNode(
	data tensor.Tensor,
	requiresGrad bool,
	op Operation,
	parents ...*Node,
) *Node {

	return &Node{

		Data: data,

		RequiresGrad: requiresGrad,

		Op: op,

		Parents: parents,
	}

}
