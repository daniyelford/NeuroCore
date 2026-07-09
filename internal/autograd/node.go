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
