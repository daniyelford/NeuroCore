package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Operation interface {
	Name() string

	Forward(
		inputs ...*Variable,
	) (*Variable, error)

	Backward(
		grad tensor.Tensor,
	) ([]tensor.Tensor, error)
}
