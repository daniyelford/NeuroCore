package autograd

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func ZeroLike(
	v Variable,
) tensor.Tensor {

	return tensor.New(
		v.Data.Shape(),
	)

}
