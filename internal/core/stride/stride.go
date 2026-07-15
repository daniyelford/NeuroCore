package stride

import "github.com/daniyelford/neurocore/internal/core/ndim"

type Stride struct {
	vector ndim.Vector
}

func New(values ...int) Stride {
	return Stride{
		vector: ndim.New(values...),
	}
}
