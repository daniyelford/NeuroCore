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

// func (s Sequential) StateDict() map[string]*autograd.Variable {

// 	out := map[string]*autograd.Variable{}

// 	for i, layer := range s.Layers {

// 		for k, v := range layer.StateDict() {

// 			out[fmt.Sprintf("%d.%s", i, k)] = v

// 		}

// 	}

//		return out
//	}
func (s *Sequential) StateDict() map[string]*autograd.Variable {

	result :=
		map[string]*autograd.Variable{}

	for index, layer := range s.Layers {

		for name, value := range layer.StateDict() {

			key :=
				fmt.Sprintf(
					"%d.%s",
					index,
					name,
				)

			result[key] = value

		}

	}

	return result
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
func (s *Sequential) Children() []Module {

	return s.Layers

}
