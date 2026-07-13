package operations

import (
	"errors"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Conv2D struct {
	Base

	StrideH int
	StrideW int

	PaddingH int
	PaddingW int

	KernelH int
	KernelW int
}

func NewConv2D(
	strideH int,
	strideW int,
	paddingH int,
	paddingW int,
	kernelH int,
	kernelW int,
) *Conv2D {

	return &Conv2D{

		StrideH: strideH,

		StrideW: strideW,

		PaddingH: paddingH,

		PaddingW: paddingW,

		KernelH: kernelH,

		KernelW: kernelW,
	}
}

func (op *Conv2D) Name() string {

	return "Conv2D"

}

func (op *Conv2D) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 3 {

		return nil, errors.New(
			"conv2d requires x weight bias",
		)
	}

	x := inputs[0]
	w := inputs[1]
	b := inputs[2]

	op.Save(
		x,
		w,
		b,
	)

	out :=
		convForward(
			x.Data(),
			w.Data(),
			b.Data(),
			op,
		)

	v :=
		autograd.NewVariable(
			out,
			x.RequiresGrad() ||
				w.RequiresGrad() ||
				b.RequiresGrad(),
		)

	v.Node().Parents =
		[]*autograd.Node{

			x.Node(),
			w.Node(),
			b.Node(),
		}

	v.Node().Op = op

	op.SetOutput(v)

	return v, nil
}

func (op *Conv2D) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	x :=
		op.Input(0).Data()

	w :=
		op.Input(1).Data()

	xShape :=
		x.Shape().Values()

	wShape :=
		w.Shape().Values()

	batch :=
		xShape[0]

	inC :=
		xShape[1]

	height :=
		xShape[2]

	width :=
		xShape[3]

	outC :=
		wShape[0]

	dx :=
		tensor.New(
			x.Shape(),
		)

	dw :=
		tensor.New(
			w.Shape(),
		)

	db :=
		tensor.New(
			shape.New(
				outC,
			),
		)

	outH :=
		grad.Shape().Values()[2]

	outW :=
		grad.Shape().Values()[3]

	for n := 0; n < batch; n++ {

		for oc := 0; oc < outC; oc++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					g :=
						grad.At(
							n,
							oc,
							oh,
							ow,
						)

					// bias gradient

					db.Set(
						db.At(oc)+g,
						oc,
					)

					for ic := 0; ic < inC; ic++ {

						for kh := 0; kh < op.KernelH; kh++ {

							for kw := 0; kw < op.KernelW; kw++ {

								ih :=
									oh*op.StrideH +
										kh -
										op.PaddingH

								iw :=
									ow*op.StrideW +
										kw -
										op.PaddingW

								if ih < 0 ||
									ih >= height ||
									iw < 0 ||
									iw >= width {

									continue

								}

								// dw

								dw.Set(
									dw.At(
										oc,
										ic,
										kh,
										kw,
									)+
										g*
											x.At(
												n,
												ic,
												ih,
												iw,
											),

									oc,
									ic,
									kh,
									kw,
								)

								// dx

								dx.Set(
									dx.At(
										n,
										ic,
										ih,
										iw,
									)+
										g*
											w.At(
												oc,
												ic,
												kh,
												kw,
											),

									n,
									ic,
									ih,
									iw,
								)

							}
						}
					}
				}
			}
		}
	}

	return []tensor.Tensor{

		dx,

		dw,

		db,
	}, nil

}
func convForward(
	x tensor.Tensor,
	w tensor.Tensor,
	b tensor.Tensor,
	op *Conv2D,
) tensor.Tensor {

	dims :=
		x.Shape().Values()

	batch :=
		dims[0]

	inC :=
		dims[1]

	height :=
		dims[2]

	width :=
		dims[3]

	wShape :=
		w.Shape().Values()

	outC :=
		wShape[0]

	outH :=
		(height+
			2*op.PaddingH-
			op.KernelH)/
			op.StrideH + 1

	outW :=
		(width+
			2*op.PaddingW-
			op.KernelW)/
			op.StrideW + 1

	out :=
		tensor.New(
			shape.New(
				batch,
				outC,
				outH,
				outW,
			),
		)

	for n := 0; n < batch; n++ {

		for oc := 0; oc < outC; oc++ {

			for oh := 0; oh < outH; oh++ {

				for ow := 0; ow < outW; ow++ {

					sum :=
						float32(0)

					for ic := 0; ic < inC; ic++ {

						for kh := 0; kh < op.KernelH; kh++ {

							for kw := 0; kw < op.KernelW; kw++ {

								ih :=
									oh*op.StrideH +
										kh -
										op.PaddingH

								iw :=
									ow*op.StrideW +
										kw -
										op.PaddingW

								if ih < 0 ||
									ih >= height ||
									iw < 0 ||
									iw >= width {

									continue

								}

								sum +=
									x.At(
										n,
										ic,
										ih,
										iw,
									) *
										w.At(
											oc,
											ic,
											kh,
											kw,
										)

							}
						}
					}

					sum += b.At(oc)

					out.Set(
						sum,
						n,
						oc,
						oh,
						ow,
					)

				}
			}
		}
	}

	return out

}
