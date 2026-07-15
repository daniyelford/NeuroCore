package trainer

import (
	"github.com/daniyelford/neurocore/dataset"
	"github.com/daniyelford/neurocore/internal/autograd"
)

func (t *Trainer) Train(
	loader dataset.DataLoader,
	epochs int,
	lossFunc func(
		*autograd.Variable,
		*autograd.Variable,
	) *autograd.Variable,
) {

	for epoch := 0; epoch < epochs; epoch++ {

		var total float32

		var count int

		for batch := range loader.Batches() {

			x :=
				autograd.NewVariable(
					batch.X,
					false,
				)

			y :=
				autograd.NewVariable(
					batch.Y,
					false,
				)

			loss :=
				t.TrainStep(
					x,
					y,
					lossFunc,
				)

			total +=
				loss.Data().Mean()

			count++

		}

		if count > 0 {

			println(
				"epoch:",
				epoch,
				"loss:",
				total/
					float32(count),
			)

		}

	}

}
