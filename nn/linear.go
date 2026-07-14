package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

type Linear struct {
	BaseModule
	Weight Parameter

	Bias Parameter

	In int

	Out int
}

func (l *Linear) Name() string {

	return "Linear"

}
func (l *Linear) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{

		"weight": l.Weight.Value,

		"bias": l.Bias.Value,
	}

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
		BaseModule: NewBaseModule(),
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

	matmul :=
		&operations.MatMul{}

	x, err :=
		matmul.Forward(
			&input,
			l.Weight.Value,
		)

	if err != nil {

		panic(err)

	}

	add :=
		&operations.Add{}

	out, err :=
		add.Forward(
			x,
			l.Bias.Value,
		)

	if err != nil {

		panic(err)

	}

	return *out

}

func (l *Linear) Parameters() []Parameter {

	return []Parameter{

		l.Weight,

		l.Bias,
	}
}
