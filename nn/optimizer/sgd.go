package optimizer

import (
	"github.com/daniyelford/neurocore/nn"
)

type SGD struct {
	LR float32
}

func NewSGD(
	lr float32,
) SGD {

	return SGD{

		LR: lr,
	}

}

func (s SGD) Step(
	params []nn.Parameter,
) {

	for _, p := range params {

		if p.Value.Grad().Empty() {

			continue

		}

		update :=
			p.Value.Grad().Scale(
				s.LR,
			)

		p.Value.Data().SubInplace(
			update,
		)

	}

}

func (s SGD) ZeroGrad(
	params []nn.Parameter,
) {

	for _, p := range params {

		p.Value.ZeroGrad()

	}

}
