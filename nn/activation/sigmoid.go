package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

type Sigmoid struct {
	nn.BaseModule
}

func NewSigmoid() *Sigmoid {

	return &Sigmoid{
		BaseModule: nn.NewBaseModule(),
	}

}

func (s *Sigmoid) Name() string {

	return "Sigmoid"

}

func (s *Sigmoid) Forward(
	input autograd.Variable,
) autograd.Variable {

	op := &operations.Sigmoid{}

	out, err :=
		op.Forward(
			&input,
		)

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
