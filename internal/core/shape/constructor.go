package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

// New creates a Shape from dimensions.
func New(dimensions ...int) Shape {
	return Shape{
		vector: ndim.New(dimensions...),
	}
}

// newFromVector creates a Shape from an existing Vector.
// It is intended for internal use only.
func newFromVector(v ndim.Vector) Shape {
	return Shape{
		vector: v,
	}
}
