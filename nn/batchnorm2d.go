package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

type BatchNorm2D struct {
	BaseModule

	Gamma Parameter

	Beta Parameter

	RunningMean tensor.Tensor

	RunningVar tensor.Tensor

	Channels int

	Eps float32

	Momentum float32
}

func NewBatchNorm2D(
	channels int,
) *BatchNorm2D {

	gamma :=
		tensor.New(
			shape.New(
				channels,
			),
		)

	beta :=
		tensor.New(
			shape.New(
				channels,
			),
		)

	return &BatchNorm2D{

		BaseModule: NewBaseModule(),

		Gamma: NewParameter(
			autograd.NewVariable(
				gamma,
				true,
			),
		),

		Beta: NewParameter(
			autograd.NewVariable(
				beta,
				true,
			),
		),

		RunningMean: tensor.New(
			shape.New(
				channels,
			),
		),

		RunningVar: tensor.New(
			shape.New(
				channels,
			),
		),

		Channels: channels,

		Eps: 1e-5,

		Momentum: 0.1,
	}

}
func (b *BatchNorm2D) Name() string {

	return "BatchNorm2D"

}
func (b *BatchNorm2D) Forward(
	input autograd.Variable,
) autograd.Variable {

	op :=
		operations.NewBatchNorm(
			b.Channels,
			b.Eps,
		)

	out, err :=
		op.Forward(
			&input,
			b.Gamma.Value,
			b.Beta.Value,
		)

	if err != nil {

		panic(err)

	}

	return *out

}
func (b *BatchNorm2D) Parameters() []Parameter {

	return []Parameter{

		b.Gamma,

		b.Beta,
	}

}
func (b *BatchNorm2D) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{

		"gamma": b.Gamma.Value,

		"beta": b.Beta.Value,
	}

}
