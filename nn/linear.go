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
) *Linear {

	w := tensor.New(
		shape.New(
			in,
			out,
		),
	)

	b := tensor.New(
		shape.New(
			out,
		),
	)
	return &Linear{
		Weight: NewParameter(
			autograd.NewVariable(
				w,
				true,
			),
		),

		Bias: NewParameter(
			autograd.NewVariable(
				b,
				true,
			),
		),

		In:  in,
		Out: out,
	}
}
func (l *Linear) Forward(
	input autograd.Variable,
) autograd.Variable {
	x, ok :=
		input.Data().MatMul(
			l.Weight.Value.Data(),
		)
	if !ok {
		panic("matmul failed")
	}
	out, ok :=
		x.AddBroadcast(
			l.Bias.Value.Data(),
		)
	if !ok {
		panic("linear bias broadcast failed")
	}
	return *autograd.NewVariable(
		out,
		true,
	)

}
func (l *Linear) Parameters() []Parameter {

	return []Parameter{

		l.Weight,

		l.Bias,
	}
}
