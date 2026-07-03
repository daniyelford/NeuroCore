package scalar

import "github.com/daniyelford/neurocore/internal/core/types"

func New[T types.Number](value T) *Scalar[T] {

	return &Scalar[T]{
		value: value,
	}

}