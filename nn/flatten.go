package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

type Flatten struct {
	BaseModule
}

func NewFlatten() *Flatten {

	return &Flatten{

		BaseModule: NewBaseModule(),
	}

}

func (f *Flatten) Name() string {

	return "Flatten"

}

func (f *Flatten) Parameters() []Parameter {

	return nil

}

func (f *Flatten) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}

func (f *Flatten) Forward(
	input autograd.Variable,
) autograd.Variable {

	d :=
		input.Data().Shape().Values()

	if len(d) < 2 {

		panic("flatten requires batch dimension")

	}

	batch :=
		d[0]

	size := 1

	for i := 1; i < len(d); i++ {

		size *= d[i]

	}

	out, ok :=
		input.Data().Reshape(
			shape.New(
				batch,
				size,
			),
		)

	if !ok {

		panic("flatten reshape failed")

	}

	return *autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}
