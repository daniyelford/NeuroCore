package dataset

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Batch struct {
	X tensor.Tensor

	Y tensor.Tensor
}
