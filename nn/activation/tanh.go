package activation

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
	"github.com/daniyelford/neurocore/nn"
)

type Tanh struct {
	nn.BaseModule
}

func NewTanh() *Tanh {

	return &Tanh{
		BaseModule: nn.NewBaseModule(),
	}

}

func (t *Tanh) Name() string {

	return "Tanh"

}

func (t *Tanh) Forward(
	input autograd.Variable,
) autograd.Variable {

	op := &operations.Tanh{}

	out, err :=
		op.Forward(
			&input,
		)

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
