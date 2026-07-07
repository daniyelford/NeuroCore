package stride

import (
	"github.com/daniyelford/neurocore/internal/core/ndim"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func New(values ...int) Stride {
	return Stride{
		vector: ndim.New(values...),
	}
}

// FromShape computes a row-major (C-order) stride from a shape.
func FromShape(s shape.Shape) Stride {

	dims := s.Values()

	if len(dims) == 0 {
		return New()
	}

	out := make([]int, len(dims))

	step := 1

	for i := len(dims) - 1; i >= 0; i-- {

		out[i] = step

		step *= dims[i]

	}

	return New(out...)

}
