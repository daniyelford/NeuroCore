package pool

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type MaxPool2D struct {
	nn.BaseModule

	KernelH int

	KernelW int

	Stride int
}

func NewMaxPool2D(
	kernelH int,
	kernelW int,
) *MaxPool2D {

	return &MaxPool2D{

		BaseModule: nn.NewBaseModule(),

		KernelH: kernelH,

		KernelW: kernelW,

		Stride: kernelH,
	}

}

func (m *MaxPool2D) Name() string {

	return "MaxPool2D"

}

func (m *MaxPool2D) Parameters() []nn.Parameter {

	return nil

}

func (m *MaxPool2D) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}

func (m *MaxPool2D) Forward(
	input autograd.Variable,
) autograd.Variable {

	in := input.Data()

	d := in.Shape().Values()

	if len(d) != 4 {

		panic("MaxPool2D expects [N,C,H,W]")

	}

	n := d[0]
	c := d[1]
	h := d[2]
	w := d[3]

	outH :=
		(h-m.KernelH)/m.Stride + 1

	outW :=
		(w-m.KernelW)/m.Stride + 1

	out :=
		tensor.New(
			shape.New(
				n,
				c,
				outH,
				outW,
			),
		)

	for b := 0; b < n; b++ {

		for ch := 0; ch < c; ch++ {

			for i := 0; i < outH; i++ {

				for j := 0; j < outW; j++ {

					max :=
						float32(-1e30)

					for kh := 0; kh < m.KernelH; kh++ {

						for kw := 0; kw < m.KernelW; kw++ {

							v :=
								in.At(
									b,
									ch,
									i*m.Stride+kh,
									j*m.Stride+kw,
								)

							if v > max {

								max = v

							}

						}

					}

					out.Set(
						max,
						b,
						ch,
						i,
						j,
					)

				}

			}

		}

	}

	return *autograd.NewVariable(
		out,
		false,
	)

}
