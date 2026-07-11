package dataset

import "github.com/daniyelford/neurocore/internal/core/tensor"

type DataLoader struct {
	dataset Dataset

	batchSize int
}

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
