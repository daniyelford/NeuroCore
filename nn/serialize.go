package nn

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type SerializedTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

type SerializedModel struct {
	Parameters map[string]SerializedTensor `json:"parameters"`
}

func serializeTensor(
	t tensor.Tensor,
) SerializedTensor {

	data := make(
		[]float32,
		t.Len(),
	)

	for i := 0; i < t.Len(); i++ {

		data[i] = t.FlatAt(i)

	}

	return SerializedTensor{

		Shape: t.Shape().Values(),

		Data: data,
	}

}

func deserializeTensor(
	s SerializedTensor,
) tensor.Tensor {

	out :=
		tensor.New(
			shape.New(
				s.Shape...,
			),
		)

	for i, v := range s.Data {

		out.FlatSet(
			i,
			v,
		)

	}

	return out

}
