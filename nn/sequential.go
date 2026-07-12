package nn

import (
	"fmt"

	"github.com/daniyelford/neurocore/internal/autograd"
)

type Sequential struct {
	Layers []Module
}

func (s Sequential) Name() string {

	return "Sequential"

}
func (s Sequential) Children() []Module {

	return s.Layers

}
func (s Sequential) StateDict() StateDict {

	out := StateDict{}

	for i, layer := range s.Layers {

		prefix :=
			fmt.Sprintf(
				"%d.",
				i,
			)

		for name, value := range layer.StateDict() {

			out[prefix+name] = value

		}

	}

	return out

}
func NewSequential(
	layers ...Module,
) Sequential {

	return Sequential{

		Layers: layers,
	}

}

func (s Sequential) Forward(
	x autograd.Variable,
) autograd.Variable {

	for _, layer := range s.Layers {

		x =
			layer.Forward(x)

	}

	return x

}
func (s Sequential) Parameters() []Parameter {

	params := []Parameter{}

	for _, layer := range s.Layers {

		params =
			append(
				params,
				layer.Parameters()...,
			)

	}

	return params

}
