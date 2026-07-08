package activation

import "github.com/daniyelford/neurocore/internal/autograd"

type Activation interface {
	Forward(
		input autograd.Variable,
	) autograd.Variable
}
