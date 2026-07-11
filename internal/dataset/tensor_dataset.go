package dataset

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type TensorDataset struct {
	X tensor.Tensor

	Y tensor.Tensor

	length int
}

func NewTensorDataset(
	x tensor.Tensor,
	y tensor.Tensor,
) TensorDataset {

	return TensorDataset{

		X: x,

		Y: y,

		length: x.Shape().Values()[0],
	}

}

func (d TensorDataset) Len() int {

	return d.length

}

func (d TensorDataset) Get(
	index int,
) (
	tensor.Tensor,
	tensor.Tensor,
) {

	x, ok :=
		d.X.Slice(
			index,
			index+1,
		)

	if !ok {

		return tensor.Tensor{}, tensor.Tensor{}

	}

	y, ok :=
		d.Y.Slice(
			index,
			index+1,
		)

	if !ok {

		return tensor.Tensor{}, tensor.Tensor{}

	}

	return x, y

}
