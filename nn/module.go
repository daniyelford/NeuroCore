package nn

import "github.com/daniyelford/neurocore/internal/autograd"

type Module interface {
	Forward(
		input autograd.Variable,
	) autograd.Variable

	Parameters() []Parameter
}
