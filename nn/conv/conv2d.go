package conv

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type Conv2D struct {
	nn.BaseModule

	Weight nn.Parameter

	Bias nn.Parameter

	InChannels int

	OutChannels int

	KernelH int

	KernelW int

	Stride int

	Padding int
}

func NewConv2D(
	inChannels int,
	outChannels int,
	kernelH int,
	kernelW int,
) *Conv2D {

	w :=
		tensor.New(
			shape.New(
				outChannels,
				inChannels,
				kernelH,
				kernelW,
			),
		)

	b :=
		tensor.New(
			shape.New(
				outChannels,
			),
		)

	return &Conv2D{

		BaseModule: nn.NewBaseModule(),

		Weight: nn.NewParameter(
			autograd.NewVariable(
				w,
				true,
			),
		),

		Bias: nn.NewParameter(
			autograd.NewVariable(
				b,
				true,
			),
		),

		InChannels: inChannels,

		OutChannels: outChannels,

		KernelH: kernelH,

		KernelW: kernelW,

		Stride: 1,

		Padding: 0,
	}
}

func (c *Conv2D) Name() string {

	return "Conv2D"

}

func (c *Conv2D) Parameters() []nn.Parameter {

	return []nn.Parameter{

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
	input autograd.Variable,
) autograd.Variable {

	x :=
		input.Data()

	d :=
		x.Shape().Values()

	if len(d) != 4 {

		panic("Conv2D expects [N,C,H,W]")

	}

	n :=
		d[0]

	h :=
		d[2]

	w :=
		d[3]

	outH :=
		(h+2*c.Padding-c.KernelH)/
			c.Stride + 1

	outW :=
		(w+2*c.Padding-c.KernelW)/
			c.Stride + 1

	out :=
		tensor.New(
			shape.New(
				n,
				c.OutChannels,
				outH,
				outW,
			),
		)

	for b := 0; b < n; b++ {

		for oc := 0; oc < c.OutChannels; oc++ {

			for oy := 0; oy < outH; oy++ {

				for ox := 0; ox < outW; ox++ {

					var sum float32

					for ic := 0; ic < c.InChannels; ic++ {

						for ky := 0; ky < c.KernelH; ky++ {

							for kx := 0; kx < c.KernelW; kx++ {

								iy :=
									oy*c.Stride + ky - c.Padding

								ix :=
									ox*c.Stride + kx - c.Padding

								if iy < 0 ||
									ix < 0 ||
									iy >= h ||
									ix >= w {

									continue
								}

								sum +=
									x.At(
										b,
										ic,
										iy,
										ix,
									) *
										c.Weight.Value.Data().At(
											oc,
											ic,
											ky,
											kx,
										)

							}
						}
					}

					sum +=
						c.Bias.Value.Data().At(oc)

					out.Set(
						sum,
						b,
						oc,
						oy,
						ox,
					)

				}
			}
		}
	}

	return *autograd.NewVariable(
		out,
		true,
	)
}
