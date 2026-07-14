package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

type BatchNorm struct {
	BaseModule

	NumFeatures int

	Eps float32

	Momentum float32

	Weight Parameter

	Bias Parameter

	RunningMean tensor.Tensor

	RunningVar tensor.Tensor
}

func NewBatchNorm(
	numFeatures int,
) *BatchNorm {

	gamma := tensor.New(
		shape.New(numFeatures),
	)
	gamma.Fill(1)

	beta := tensor.New(
		shape.New(numFeatures),
	)

	runningMean := tensor.New(
		shape.New(numFeatures),
	)

	runningVar := tensor.New(
		shape.New(numFeatures),
	)
	runningVar.Fill(1)

	return &BatchNorm{

		BaseModule: NewBaseModule(),

		NumFeatures: numFeatures,

		Eps: 1e-5,

		Momentum: 0.1,

		Weight: NewParameter(
			autograd.NewVariable(
				gamma,
				true,
			),
		),

		Bias: NewParameter(
			autograd.NewVariable(
				beta,
				true,
			),
		),

		RunningMean: runningMean,

		RunningVar: runningVar,
	}

}

func (b *BatchNorm) Name() string {

	return "BatchNorm"

}

func (b *BatchNorm) Parameters() []Parameter {

	return []Parameter{

		b.Weight,

		b.Bias,
	}

}

func (b *BatchNorm) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{

		"weight": b.Weight.Value,

		"bias": b.Bias.Value,
	}

}
func (b *BatchNorm) Forward(
	input autograd.Variable,
) autograd.Variable {

	op :=
		operations.NewBatchNorm(
			b.NumFeatures,
			b.Eps,
		)

	out, err :=
		op.Forward(
			&input,
			b.Weight.Value,
			b.Bias.Value,
		)

	if err != nil {

		panic(err)

	}

	return *out

}

// func (b *BatchNorm) Forward(
// 	input autograd.Variable,
// ) autograd.Variable {

// 	x := input.Data()

// 	out := tensor.New(
// 		x.Shape(),
// 	)

// 	for i := 0; i < x.Len(); i++ {

// 		v := x.FlatAt(i)

// 		mean := b.RunningMean.FlatAt(i % b.NumFeatures)

// 		variance := b.RunningVar.FlatAt(i % b.NumFeatures)

// 		gamma := b.Weight.Value.Data().FlatAt(i % b.NumFeatures)

// 		beta := b.Bias.Value.Data().FlatAt(i % b.NumFeatures)

// 		y :=
// 			((v-mean)/
// 				float32(math.Sqrt(float64(variance+b.Eps))))*
// 				gamma +
// 				beta

// 		out.FlatSet(
// 			i,
// 			y,
// 		)

// 	}

// 	return *autograd.NewVariable(
// 		out,
// 		input.RequiresGrad(),
// 	)

// }
