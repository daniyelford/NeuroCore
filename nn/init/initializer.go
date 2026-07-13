package init

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Initializer interface {
	Init(
		t *tensor.Tensor,
	)
}
