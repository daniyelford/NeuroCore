package examples

import (
	"testing"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
	"github.com/daniyelford/neurocore/training"
)

func TestTrainerStep(
	t *testing.T,
) {

	layer :=
		nn.NewLinear(
			2,
			2,
		)

	optimizer :=
		optim.NewSGD(
			layer.Parameters(),
			0.1,
		)

	loss :=
		nn.NewCrossEntropyLoss()

	trainer :=
		training.NewTrainer(
			layer,
			optimizer,
			loss,
		)

	inputTensor :=
		tensor.New(
			shape.New(
				1,
				2,
			),
		)

	inputTensor.Set(
		1,
		0,
		0,
	)

	inputTensor.Set(
		2,
		0,
		1,
	)

	targetTensor :=
		tensor.New(
			shape.New(1),
		)

	targetTensor.Set(
		0,
		0,
	)

	input :=
		autograd.NewVariable(
			inputTensor,
			false,
		)

	target :=
		autograd.NewVariable(
			targetTensor,
			false,
		)

	before :=
		layer.Weight.Value.Data().FlatAt(0)

	lossValue :=
		trainer.TrainStep(
			*input,
			*target,
		)

	if lossValue <= 0 {

		t.Fatalf(
			"invalid loss",
		)

	}

	after :=
		layer.Weight.Value.Data().FlatAt(0)

	if before == after {

		t.Fatalf(
			"weights did not update",
		)

	}

}
