package normalization

import (
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type LayerNorm struct {
	nn.BaseModule
	Shape shape.Shape
	Gamma nn.Parameter
	Beta  nn.Parameter
	Eps   float32
}

func New(
	features int,
) LayerNorm {

	gamma :=
		tensor.New(
			shape.New(features),
		)

	beta :=
		tensor.New(
			shape.New(features),
		)

	return LayerNorm{

		BaseModule: nn.NewBaseModule(),

		Shape: shape.New(features),

		Gamma: nn.NewParameter(
			autograd.NewVariable(
				gamma,
				true,
			),
		),

		Beta: nn.NewParameter(
			autograd.NewVariable(
				beta,
				true,
			),
		),

		Eps: 1e-5,
	}

}
func (l LayerNorm) Forward(
	input autograd.Variable,
) autograd.Variable {

	x :=
		input.Data()

	var mean float32

	for i := 0; i < x.Len(); i++ {

		mean += x.FlatAt(i)

	}

	mean /= float32(x.Len())

	var variance float32

	for i := 0; i < x.Len(); i++ {

		diff :=
			x.FlatAt(i) - mean

		variance += diff * diff

	}

	variance /= float32(x.Len())

	out :=
		tensor.New(
			x.Shape(),
		)

	for i := 0; i < x.Len(); i++ {

		n :=
			(x.FlatAt(i) - mean) /
				float32(
					math.Sqrt(
						float64(variance+l.Eps),
					),
				)

		v :=
			n*l.Gamma.Value.Data().FlatAt(i) +
				l.Beta.Value.Data().FlatAt(i)

		out.FlatSet(
			i,
			v,
		)

	}

	return *autograd.NewVariable(
		out,
		true,
	)

}
func (l LayerNorm) Parameters() []nn.Parameter {

	return []nn.Parameter{

		l.Gamma,

		l.Beta,
	}

}
