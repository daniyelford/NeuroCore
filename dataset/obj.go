package dataset

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type JSONSample struct {
	X []float32 `json:"x"`

	Y int `json:"y"`
}

type JSONFile struct {
	Samples []JSONSample `json:"samples"`
}
type Dataset interface {
	Len() int

	Get(index int) (
		tensor.Tensor,
		tensor.Tensor,
	)
}
type Batch struct {
	X tensor.Tensor

	Y tensor.Tensor
}
type DataLoader struct {
	dataset Dataset

	batchSize int
}
type TensorDataset struct {
	X tensor.Tensor

	Y tensor.Tensor

	length int
}
