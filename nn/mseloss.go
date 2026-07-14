package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
)

type MSELoss struct {
	BaseModule
}

func NewMSELoss() *MSELoss {

	return &MSELoss{

		BaseModule: NewBaseModule(),
	}

}

func (m *MSELoss) Name() string {

	return "MSELoss"

}

func (m *MSELoss) Parameters() []Parameter {

	return nil

}

func (m *MSELoss) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}

func (m *MSELoss) Forward(
	prediction autograd.Variable,
	target autograd.Variable,
) autograd.Variable {

	op := &operations.MSELoss{}

	out, err := op.Forward(

		&prediction,

		&target,
	)

	if err != nil {

		panic(err)

	}

	return *out

}
