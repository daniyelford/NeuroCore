package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

// New creates a new Shape.
func New(dimensions ...int) Shape {
	return Shape{
		vector: ndim.New(dimensions...),
	}
}
