package scalar

import "github.com/daniyalfrd/neurocore/internal/core/types"

func New[T types.Number](v T) *Scalar[T] {
	return &Scalar[T]{
		value: v,
	}
}