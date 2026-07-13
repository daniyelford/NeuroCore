package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

type ReLU struct {
	nn.BaseModule
}

func NewReLU() *ReLU {

	return &ReLU{
		BaseModule: nn.NewBaseModule(),
	}

}

func (r *ReLU) Name() string {

	return "ReLU"

}

func (r *ReLU) Forward(
	input autograd.Variable,
) autograd.Variable {

	op := &operations.ReLU{}

	out, err :=
		op.Forward(
			&input,
		)

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
