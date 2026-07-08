package loss

import "github.com/daniyelford/neurocore/internal/autograd"

type Loss interface {
	Forward(
		prediction autograd.Variable,
		target autograd.Variable,
	) autograd.Variable
}
