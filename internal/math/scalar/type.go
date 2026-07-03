package scalar

import "github.com/daniyelford/neurocore/internal/core/types"

type Scalar[T types.Number] struct {
	value T
}