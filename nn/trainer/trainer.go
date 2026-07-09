package trainer

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
)

func TrainStep(
	model *nn.Model,
	input autograd.Variable,
	target autograd.Variable,
	lossFunc func(
		autograd.Variable,
		autograd.Variable,
	) autograd.Variable,
) {

	output :=
		model.Root.Forward(
			input,
		)

	loss :=
		lossFunc(
			output,
			target,
		)

	autograd.Backward(
		loss,
	)

}
