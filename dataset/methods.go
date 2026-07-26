/*
Package dataset provides dataset loaders.
*/
package dataset

import (
	"encoding/json"
	"os"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func NewDataLoader(
	d Dataset,
	batchSize int,
) DataLoader {

	return DataLoader{

		dataset: d,

		batchSize: batchSize,
	}

}
func (l DataLoader) Batches() <-chan Batch {

	ch :=
		make(chan Batch)

	go func() {

		defer close(ch)

		for start := 0; start < l.dataset.Len(); start += l.batchSize {

			end :=
				start + l.batchSize

			if end > l.dataset.Len() {

				end =
					l.dataset.Len()

			}

			var xs []tensor.Tensor
			var ys []tensor.Tensor

			for i := start; i < end; i++ {

				x, y :=
					l.dataset.Get(i)

				xs =
					append(xs, x)

				ys =
					append(ys, y)

			}
			bx, _ :=
				tensor.Stack(xs)

			by, _ :=
				tensor.Stack(ys)
			ch <- Batch{

				X: bx,

				Y: by,
			}

		}

	}()

	return ch

}
func LoadJSON(
	path string,
) (
	tensor.Tensor,
	tensor.Tensor,
	error,
) {

	bytes, err :=
		os.ReadFile(path)

	if err != nil {

		return tensor.Tensor{},
			tensor.Tensor{},
			err

	}

	var file JSONFile

	err =
		json.Unmarshal(
			bytes,
			&file,
		)

	if err != nil {

		return tensor.Tensor{},
			tensor.Tensor{},
			err

	}

	rows :=
		len(file.Samples)

	cols :=
		len(file.Samples[0].X)

	x :=
		tensor.New(
			shape.New(
				rows,
				cols,
			),
		)

	y :=
		tensor.New(
			shape.New(
				rows,
			),
		)

	for i, s := range file.Samples {

		for j, v := range s.X {

			x.Set(
				v,
				i,
				j,
			)

		}

		y.Set(
			float32(s.Y),
			i,
		)

	}

	return x, y, nil

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
func (d TensorDataset) Get(
	index int,
) (
	tensor.Tensor,
	tensor.Tensor,
) {

	x, _ :=
		d.X.Slice(
			index,
			index+1,
		)

	y, _ :=
		d.Y.Slice(
			index,
			index+1,
		)

	return x.Squeeze(), y.Squeeze()

}
func (d TensorDataset) Len() int {

	return d.length

}
