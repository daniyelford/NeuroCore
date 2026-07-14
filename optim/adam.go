package optim

import (
	"math"

	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type Adam struct {
	Parameters []nn.Parameter

	LR float32

	Beta1 float32

	Beta2 float32

	Eps float32

	M []tensor.Tensor

	V []tensor.Tensor

	StepCount int
}

func NewAdam(
	parameters []nn.Parameter,
	lr float32,
) *Adam {

	m :=
		make(
			[]tensor.Tensor,
			len(parameters),
		)

	v :=
		make(
			[]tensor.Tensor,
			len(parameters),
		)

	for i, p := range parameters {

		m[i] =
			tensor.New(
				p.Value.Data().Shape(),
			)

		v[i] =
			tensor.New(
				p.Value.Data().Shape(),
			)

	}

	return &Adam{

		Parameters: parameters,

		LR: lr,

		Beta1: 0.9,

		Beta2: 0.999,

		Eps: 1e-8,

		M: m,

		V: v,
	}

}
func (a *Adam) Step() {

	a.StepCount++

	for i, p := range a.Parameters {

		data :=
			p.Value.Data()

		grad :=
			p.Value.Grad()

		m :=
			a.M[i]

		v :=
			a.V[i]

		for j := 0; j < data.NumElements(); j++ {

			g :=
				grad.FlatAt(j)

			mValue :=
				a.Beta1*m.FlatAt(j) +
					(1-a.Beta1)*g

			vValue :=
				a.Beta2*v.FlatAt(j) +
					(1-a.Beta2)*g*g

			m.FlatSet(
				j,
				mValue,
			)

			v.FlatSet(
				j,
				vValue,
			)

			mHat :=
				mValue /
					float32(
						1-math.Pow(
							float64(a.Beta1),
							float64(a.StepCount),
						),
					)

			vHat :=
				vValue /
					float32(
						1-math.Pow(
							float64(a.Beta2),
							float64(a.StepCount),
						),
					)

			update :=
				a.LR *
					mHat /
					(float32(
						math.Sqrt(float64(vHat))) + a.Eps)

			data.FlatSet(
				j,
				data.FlatAt(j)-update,
			)

		}

		a.M[i] = m

		a.V[i] = v

	}

}
func (a *Adam) ZeroGrad() {

	for _, p := range a.Parameters {

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
