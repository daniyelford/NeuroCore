package training

import (
	"fmt"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
)

type Trainable interface {
	Forward(
		input autograd.Variable,
	) autograd.Variable

	Parameters() []nn.Parameter
}

type Trainer struct {
	Model Trainable

	Optimizer optim.Optimizer

	Loss *nn.CrossEntropyLoss
}

func NewTrainer(
	model Trainable,
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
	for _, p := range t.Model.Parameters() {

		fmt.Println("print in trainer",
			p.Value.Grad().NumElements(),
		)

	}
	t.Optimizer.Step()

	return loss.Data().FlatAt(0)

}
