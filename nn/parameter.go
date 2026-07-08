package nn

import "github.com/daniyelford/neurocore/internal/autograd"

type Parameter struct {
	Value autograd.Variable
}

func NewParameter(
	v autograd.Variable,
) Parameter {

	return Parameter{

		Value: v,
	}

}
