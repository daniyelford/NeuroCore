package nn

import "github.com/daniyelford/neurocore/internal/autograd"

type Sequential struct {
	Layers []Module
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
