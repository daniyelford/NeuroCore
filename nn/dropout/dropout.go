package dropout

import (
	"math/rand"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type Dropout struct {
	nn.BaseModule

	P float32
}

func New(
	p float32,
) Dropout {

	return Dropout{

		BaseModule: nn.NewBaseModule(),

		P: p,
	}

}

func (d Dropout) Forward(
	input autograd.Variable,
) autograd.Variable {

	// evaluation mode
	if !d.Training() {

		return input

	}

	mask :=
		tensor.New(
			input.Data().Shape(),
		)

	for i := 0; i < mask.Len(); i++ {

		if rand.Float32() > d.P {

			mask.FlatSet(
				i,
				1,
			)

		} else {

			mask.FlatSet(
				i,
				0,
			)

		}

	}

	out :=
		input.Data().Mul(
			mask,
		)

	return *autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}

func (d Dropout) Parameters() []nn.Parameter {

	return nil

}

func (d Dropout) Name() string {

	return "Dropout"

}

func (d Dropout) Children() []nn.Module {

	return nil

}
