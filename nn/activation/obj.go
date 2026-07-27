package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/nn"
)

type Activation interface {
	Forward(input *autograd.Variable) *autograd.Variable
}
type ReLU struct {
	nn.BaseModule
}
type Sigmoid struct {
	nn.BaseModule
}
type Tanh struct {
	nn.BaseModule
}
