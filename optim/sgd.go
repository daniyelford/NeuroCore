package optim

import "github.com/daniyelford/neurocore/nn"

type SGD struct {
	Parameters []nn.Parameter

	LR float32
}

func NewSGD(
	parameters []nn.Parameter,
	lr float32,
) *SGD {

	return &SGD{

		Parameters: parameters,

		LR: lr,
	}

}
func (s *SGD) Step() {

	for _, p := range s.Parameters {

		if p.Value == nil {

			continue

		}

		data :=
			p.Value.Data()

		grad :=
			p.Value.Grad()

		for i := 0; i < data.NumElements(); i++ {

			value :=
				data.FlatAt(i)

			g :=
				grad.FlatAt(i)

			data.FlatSet(
				i,
				value-s.LR*g,
			)

		}

	}

}
func (s *SGD) ZeroGrad() {

	for _, p := range s.Parameters {

		if p.Value == nil {

			continue

		}

		grad :=
			p.Value.Grad()

		for i := 0; i < grad.NumElements(); i++ {

			grad.FlatSet(
				i,
				0,
			)

		}

	}

}
