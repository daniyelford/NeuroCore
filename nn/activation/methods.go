package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

func NewReLU() *ReLU {
	return &ReLU{
		BaseModule: nn.NewBaseModule("ReLU"),
	}
}
func (r *ReLU) Forward(input autograd.Variable) autograd.Variable {
	op := &operations.ReLU{}
	out, err := op.Forward(&input)
	if err != nil {
		panic(err)
	}
	return *out
}
func (r *ReLU) Parameters() []nn.Parameter {
	return nil
}
func (r *ReLU) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewSigmoid() *Sigmoid {
	return &Sigmoid{
		BaseModule: nn.NewBaseModule("Sigmoid"),
	}
}
func (s *Sigmoid) Forward(input autograd.Variable) autograd.Variable {
	op := &operations.Sigmoid{}
	out, err := op.Forward(&input)
	if err != nil {
		panic(err)
	}
	return *out
}
func (s *Sigmoid) Parameters() []nn.Parameter {
	return nil
}
func (s *Sigmoid) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
func NewTanh() *Tanh {
	return &Tanh{
		BaseModule: nn.NewBaseModule("Tanh"),
	}
}
func (t *Tanh) Forward(input autograd.Variable) autograd.Variable {
	op := &operations.Tanh{}
	out, err := op.Forward(&input)
	if err != nil {
		panic(err)
	}
	return *out
}
func (t *Tanh) Parameters() []nn.Parameter {
	return nil
}
func (t *Tanh) StateDict() map[string]*autograd.Variable {
	return map[string]*autograd.Variable{}
}
