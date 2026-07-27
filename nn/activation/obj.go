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
type LeakyReLU struct {
	nn.BaseModule
	Alpha float32
}
type ELU struct {
	nn.BaseModule
	Alpha float32
}
type GELU struct {
	nn.BaseModule
}
type Softplus struct {
	nn.BaseModule
}
type Swish struct {
	nn.BaseModule
}
type Mish struct {
	nn.BaseModule
}
type HardSigmoid struct {
	nn.BaseModule
}
type HardSwish struct {
	nn.BaseModule
}
type Softmax struct {
	nn.BaseModule
	Axis int
}
type LogSoftmax struct {
	nn.BaseModule
	Axis int
}
