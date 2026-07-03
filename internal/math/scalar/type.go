package scalar

import "github.com/daniyalfrd/neurocore/internal/core/types"

type Scalar[T types.Number] struct {
	value T
}