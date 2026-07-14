package optim

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type Momentum struct {
	Parameters []nn.Parameter

	LR float32

	Beta float32

	Velocity []tensor.Tensor
}

func NewMomentum(
	parameters []nn.Parameter,
	lr float32,
	beta float32,
) *Momentum {

	velocity :=
		make(
			[]tensor.Tensor,
			len(parameters),
		)

	for i, p := range parameters {

		velocity[i] =
			tensor.New(
				p.Value.Data().Shape(),
			)

	}

	return &Momentum{

		Parameters: parameters,

		LR: lr,

		Beta: beta,

		Velocity: velocity,
	}

}
func (m *Momentum) Step() {

	for i, p := range m.Parameters {

		data :=
			p.Value.Data()

		grad :=
			p.Value.Grad()

		v :=
			m.Velocity[i]

		for j := 0; j < data.NumElements(); j++ {

			old :=
				v.FlatAt(j)

			newV :=
				m.Beta*old +
					grad.FlatAt(j)

			v.FlatSet(
				j,
				newV,
			)

			data.FlatSet(
				j,
				data.FlatAt(j)-
					m.LR*newV,
			)

		}

		m.Velocity[i] = v

	}

}
func (m *Momentum) ZeroGrad() {

	for _, p := range m.Parameters {

		g :=
			p.Value.Grad()

		for i := 0; i < g.NumElements(); i++ {

			g.FlatSet(
				i,
				0,
			)

		}

	}

}
