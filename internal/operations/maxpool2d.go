package operations

import (
	"errors"
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type MaxPool2D struct {
	Base

	KernelH int
	KernelW int

	StrideH int
	StrideW int

	ArgMax []int
}

func NewMaxPool2D(
	kernelH,
	kernelW,
	strideH,
	strideW int,
) *MaxPool2D {

	return &MaxPool2D{

		KernelH: kernelH,
		KernelW: kernelW,

		StrideH: strideH,
		StrideW: strideW,
	}

}

func (op *MaxPool2D) Name() string {

	return "MaxPool2D"

}

func (op *MaxPool2D) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 1 {

		return nil,
			errors.New(
				"maxpool requires one input",
			)

	}

	x := inputs[0]

	op.Save(x)

	dims := x.Data().Shape().Values()

	batch := dims[0]
	channels := dims[1]
	height := dims[2]
	width := dims[3]

	outH :=
		(height-op.KernelH)/op.StrideH + 1

	outW :=
		(width-op.KernelW)/op.StrideW + 1

	out :=
		tensor.New(
			shape.New(
				batch,
				channels,
				outH,
				outW,
			),
		)

	count :=
		batch *
			channels *
			outH *
			outW

	op.ArgMax =
		make(
			[]int,
			count,
		)

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					maxValue :=
						float32(
							math.Inf(-1),
						)

					maxIndex := 0

					for kh := 0; kh < op.KernelH; kh++ {

						for kw := 0; kw < op.KernelW; kw++ {

							ih :=
								oh*op.StrideH +
									kh

							iw :=
								ow*op.StrideW +
									kw

							v :=
								x.Data().At(
									n,
									c,
									ih,
									iw,
								)

							if v > maxValue {

								maxValue = v

								maxIndex =
									((n*channels+c)*
										height+
										ih)*
										width +
										iw

							}

						}

					}

					out.Set(
						maxValue,
						n,
						c,
						oh,
						ow,
					)

					outputIndex :=
						(((n*channels+c)*
							outH+
							oh)*
							outW +
							ow)

					op.ArgMax[outputIndex] =
						maxIndex

				}

			}

		}

	}

	v :=
		autograd.NewVariable(
			out,
			x.RequiresGrad(),
		)

	v.Node().Parents =
		[]*autograd.Node{

			x.Node(),
		}

	v.Node().Op = op

	op.SetOutput(v)

	return v, nil

}
func (op *MaxPool2D) Backward(
	grad tensor.Tensor,
) (
	[]tensor.Tensor,
	error,
) {

	input := op.Input(0).Data()

	dx := tensor.New(
		input.Shape(),
	)

	dims := grad.Shape().Values()

	batch := dims[0]
	channels := dims[1]
	outH := dims[2]
	outW := dims[3]

	index := 0

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for h := 0; h < outH; h++ {

				for w := 0; w < outW; w++ {

					inputIndex :=
						op.ArgMax[index]

					old :=
						dx.FlatAt(
							inputIndex,
						)

					dx.FlatSet(
						inputIndex,
						old+
							grad.At(
								n,
								c,
								h,
								w,
							),
					)

					index++

				}

			}

		}

	}

	return []tensor.Tensor{
		dx,
	}, nil

}
