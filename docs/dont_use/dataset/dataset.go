package dataset

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Dataset interface {
	Len() int

	Get(index int) (
		tensor.Tensor,
		tensor.Tensor,
	)
}
