package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Linear struct {
	Weight Parameter

	Bias Parameter

	In int

	Out int
}

func NewLinear(
	in int,
	out int,
) Linear {

	w :=
		tensor.New(
			shape.New(
				in,
				out,
			),
		)

	b :=
		tensor.New(
			shape.New(
				out,
			),
		)
	wv := autograd.NewVariable(w, true)
	bv := autograd.NewVariable(b, true)
	return Linear{
		Weight: NewParameter(&wv),
		Bias:   NewParameter(&bv),
		In:     in,
		Out:    out,
	}

}
func (l Linear) Forward(
	input autograd.Variable,
) autograd.Variable {

	x, _ :=
		input.Data.MatMul(
			l.Weight.Value.Data,
		)

	out :=
		x.Add(
			l.Bias.Value.Data,
		)

	return autograd.NewVariable(
		out,
		true,
	)

}
func (l Linear) Parameters() []Parameter {

	return []Parameter{

		l.Weight,

		l.Bias,
	}

}
