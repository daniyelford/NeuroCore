package trainer

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/nn/optimizer"
)

type Trainer struct {
	Model     *nn.Model
	Optimizer optimizer.Optimizer
}

func NewTrainer(
	model *nn.Model,
	opt optimizer.Optimizer,
) *Trainer {

	return &Trainer{

		Model: model,

		Optimizer: opt,
	}

}
func (t *Trainer) TrainStep(
	input *autograd.Variable,
	target *autograd.Variable,
	lossFunc func(
		*autograd.Variable,
		*autograd.Variable,
	) *autograd.Variable,
) *autograd.Variable {

	params :=
		t.Model.Parameters()

	t.Optimizer.ZeroGrad(
		params,
	)

	output :=
		t.Model.Forward(
			*input,
		)

	loss :=
		lossFunc(
			&output,
			target,
		)

	autograd.Backward(
		loss,
	)

	t.Optimizer.Step(
		params,
	)

	return loss

}
