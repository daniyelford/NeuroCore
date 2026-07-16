package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/daniyelford/neurocore/dataset"
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
	"github.com/daniyelford/neurocore/training"
)

func main() {
	path := "examples/json_classifier/data.json"
	count := 1000
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	if len(os.Args) > 2 {
		arg := os.Args[2]
		n, err := strconv.Atoi(arg)
		if err != nil {
			fmt.Println("arg 2 must be a number", err)
			return
		}
		count = n
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
	lossFunc := nn.NewCrossEntropyLoss()
	t :=
		training.NewTrainer(
			model,
			optimizer,
			lossFunc,
		)

	for epoch := range count {

		var total float32
		for batch := range loader.Batches() {
			x := autograd.NewVariable(
				batch.X,
				false,
			)
			y := autograd.NewVariable(
				batch.Y,
				false,
			)
			loss :=
				t.TrainStep(
					*x,
					*y,
				)

			total += loss
		}
		if epoch > 0 {
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
