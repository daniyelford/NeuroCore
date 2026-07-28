package training

import (
	"github.com/daniyelford/neurocore/dataset"
	"github.com/daniyelford/neurocore/nn"
	"github.com/daniyelford/neurocore/optim"
)

// type Trainable interface {
// 	Forward(
// 		input *autograd.Variable,
// 	) *autograd.Variable

// 	Parameters() []nn.Parameter
// }

type Trainer struct {
	Model nn.Module

	Optimizer optim.Optimizer

	Loss *nn.CrossEntropyLoss
}
type Loader interface {
	Batches() <-chan dataset.Batch
}
type Epoch struct {
	Number int
	Train  struct {
		Loss     float32
		Accuracy float32
	}
	Validation struct {
		Loss     float32
		Accuracy float32
	}
	LearningRate float32
}

type History struct {
	Epochs []Epoch
}
