package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

type Conv2D struct {
	BaseModule

	InChannels int

	OutChannels int

	KernelH int

	KernelW int

	StrideH int

	StrideW int

	PaddingH int

	PaddingW int

	Weight Parameter

	Bias Parameter
}

func NewConv2D(
	inChannels int,
	outChannels int,
	kernelH int,
	kernelW int,
	stride int,
	padding int,
) *Conv2D {

	weight :=
		tensor.New(
			shape.New(
				outChannels,
				inChannels,
				kernelH,
				kernelW,
			),
		)

	bias :=
		tensor.New(
			shape.New(
				outChannels,
			),
		)

	return &Conv2D{

		BaseModule: NewBaseModule(),

		InChannels: inChannels,

		OutChannels: outChannels,

		KernelH: kernelH,

		KernelW: kernelW,

		StrideH: stride,

		StrideW: stride,

		PaddingH: padding,

		PaddingW: padding,

		Weight: NewParameter(
			autograd.NewVariable(
				weight,
				true,
			),
		),

		Bias: NewParameter(
			autograd.NewVariable(
				bias,
				true,
			),
		),
	}
}
func (c *Conv2D) Name() string {

	return "Conv2D"

}

func (c *Conv2D) Parameters() []Parameter {

	return []Parameter{

		c.Weight,

		c.Bias,
	}
}

func (c *Conv2D) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{

		"weight": c.Weight.Value,

		"bias": c.Bias.Value,
	}
}
func (c *Conv2D) Forward(
	input *autograd.Variable,
) *autograd.Variable {

	op :=
		operations.NewConv2D(
			c.StrideH,
			c.StrideW,
			c.PaddingH,
			c.PaddingW,
			c.KernelH,
			c.KernelW,
		)

	out, err :=
		op.Forward(
			input,
			c.Weight.Value,
			c.Bias.Value,
		)

	if err != nil {
		panic(err)
	}

	return out
}
