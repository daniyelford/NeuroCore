package optimizer

import "github.com/daniyelford/neurocore/nn"

type Optimizer interface {
	Step(
		params []nn.Parameter,
	)

	ZeroGrad(
		params []nn.Parameter,
	)
}
