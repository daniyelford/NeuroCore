package main

import (
	"fmt"
	"os"

	"github.com/daniyelford/neurocore/dataset"
	"github.com/daniyelford/neurocore/docs/dont_use/training"
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
)

func main() {
	path := "examples/json_classifier/data.json"

	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	x, y, err :=
		dataset.LoadJSON(path)

	if err != nil {

		panic(err)

	}

	ds :=
		dataset.NewTensorDataset(
			x,
			y,
		)

	loader :=
		dataset.NewDataLoader(
			ds,
			2,
		)

	model :=
		nn.NewLinear(
			2,
			2,
		)

	optimizer :=
		optim.NewSGD(
			model.Parameters(),
			0.1,
		)

	loss :=
		nn.NewCrossEntropyLoss()

	trainer :=
		training.NewTrainer(
			model,
			optimizer,
			loss,
		)

	for epoch := 0; epoch < 50; epoch++ {

		total :=
			float32(0)

		for batch := range loader.Batches() {

			input :=
				autograd.NewVariable(
					batch.X,
					false,
				)

			target :=
				autograd.NewVariable(
					batch.Y,
					false,
				)

			l :=
				trainer.TrainStep(
					*input,
					*target,
				)

			total += l

		}

		fmt.Println(
			"epoch",
			epoch,
			"loss",
			total,
		)

	}

}
