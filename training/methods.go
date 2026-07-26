package training

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
)

func NewTrainer(
	model nn.Module,
	optimizer optim.Optimizer,
	loss *nn.CrossEntropyLoss,
) *Trainer {

	return &Trainer{

		Model: model,

		Optimizer: optimizer,

		Loss: loss,
	}

}

func (t *Trainer) TrainStep(
	input autograd.Variable,
	target autograd.Variable,
) float32 {

	t.Optimizer.ZeroGrad()

	prediction :=
		t.Model.Forward(
			input,
		)

	loss :=
		t.Loss.Forward(
			prediction,
			target,
		)

	autograd.Backward(
		&loss,
	)
	t.Optimizer.Step()

	return loss.Data().FlatAt(0)

}
func (t *Trainer) Run(
	loader Loader,
	epochs int,
) History {

	history := History{}

	for epoch := 0; epoch < epochs; epoch++ {

		var total float32
		batchCount := 0

		for batch := range loader.Batches() {

			x := autograd.NewVariable(
				batch.X,
				false,
			)

			y := autograd.NewVariable(
				batch.Y,
				false,
			)

			loss := t.TrainStep(
				*x,
				*y,
			)

			total += loss
			batchCount++
		}
		var avgLoss float32
		if batchCount > 0 {
			avgLoss = total / float32(batchCount)
		}
		e := Epoch{}

		e.Number = epoch + 1

		e.Train.Loss = avgLoss
		history.Epochs = append(
			history.Epochs,
			e,
		)

		println(
			"epoch:",
			e.Number,
			"loss:",
			e.Train.Loss,
		)
	}
	return history
}
