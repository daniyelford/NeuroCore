package optimizer

import (
	"github.com/daniyelford/neurocore/internal/core/nn"
)

type Optimizer interface {
	Step(
		params []nn.Parameter,
	)

	ZeroGrad(
		params []nn.Parameter,
	)
}
