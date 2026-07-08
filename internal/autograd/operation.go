package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Operation interface {
	Forward() Variable

	Backward(
		grad tensor.Tensor,
	) []tensor.Tensor

	Name() string
}
