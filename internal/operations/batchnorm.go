package operations

import (
	"errors"
	"math"

	"github.com/daniyelford/neurocore/internal/autograd"
	shapepkg "github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type BatchNorm struct {
	Base

	Eps float32

	Channels int

	Mean tensor.Tensor

	Variance tensor.Tensor
}

func NewBatchNorm(
	channels int,
	eps float32,
) *BatchNorm {

	return &BatchNorm{

		Channels: channels,

		Eps: eps,
	}

}

func (op *BatchNorm) Name() string {

	return "BatchNorm"

}
func (op *BatchNorm) Forward(
	inputs ...*autograd.Variable,
) (*autograd.Variable, error) {

	if len(inputs) != 3 {

		return nil,
			errors.New(
				"batchnorm requires input gamma beta",
			)

	}

	x :=
		inputs[0]

	gamma :=
		inputs[1]

	beta :=
		inputs[2]

	op.Save(
		x,
		gamma,
		beta,
	)

	out :=
		tensor.New(
			x.Data().Shape(),
		)

	shape :=
		x.Data().Shape().Values()

	batch :=
		shape[0]

	channels :=
		shape[1]

	height :=
		shape[2]

	width :=
		shape[3]

	mean :=
		make([]float32, channels)

	variance :=
		make([]float32, channels)

	size :=
		float32(
			batch * height * width,
		)

	// mean

	for c := 0; c < channels; c++ {

		sum := float32(0)

		for n := 0; n < batch; n++ {

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					sum +=
						x.Data().At(
							n,
							c,
							h,
							w,
						)

				}
			}
		}

		mean[c] = sum / size

	}

	// variance

	for c := 0; c < channels; c++ {

		sum := float32(0)

		for n := 0; n < batch; n++ {

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					v :=
						x.Data().At(
							n,
							c,
							h,
							w,
						) -
							mean[c]

					sum += v * v

				}
			}
		}

		variance[c] = sum / size

	}

	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					v :=
						x.Data().At(
							n,
							c,
							h,
							w,
						)

					norm :=
						(v - mean[c]) /
							float32(
								math.Sqrt(
									float64(
										variance[c]+op.Eps,
									),
								),
							)

					y :=
						norm*
							gamma.Data().At(c) +
							beta.Data().At(c)

					out.Set(
						y,
						n,
						c,
						h,
						w,
					)

				}
			}
		}
	}

	op.Mean =
		tensor.New(
			shapepkg.New(
				channels,
			),
		)

	op.Variance =
		tensor.New(
			shapepkg.New(
				channels,
			),
		)

	v :=
		autograd.NewVariable(
			out,
			x.RequiresGrad() ||
				gamma.RequiresGrad() ||
				beta.RequiresGrad(),
		)

	v.Node().Parents =
		[]*autograd.Node{

			x.Node(),

			gamma.Node(),

			beta.Node(),
		}

	v.Node().Op = op

	op.SetOutput(v)

	return v, nil

}
func (op *BatchNorm) Backward(
	grad tensor.Tensor,
) ([]tensor.Tensor, error) {

	x :=
		op.Input(0).Data()

	gamma :=
		op.Input(1).Data()

	dx :=
		tensor.New(
			x.Shape(),
		)

	dgamma :=
		tensor.New(
			gamma.Shape(),
		)

	dbeta :=
		tensor.New(
			gamma.Shape(),
		)

	dims :=
		x.Shape().Values()

	batch :=
		dims[0]

	channels :=
		dims[1]

	height :=
		dims[2]

	width :=
		dims[3]

	size :=
		float32(
			batch *
				height *
				width,
		)
	for c := 0; c < channels; c++ {

		sum :=
			float32(0)

		for n := 0; n < batch; n++ {

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					sum +=
						grad.At(
							n,
							c,
							h,
							w,
						)

				}
			}
		}

		dbeta.Set(
			sum,
			c,
		)

	}
	for c := 0; c < channels; c++ {

		sum :=
			float32(0)

		for n := 0; n < batch; n++ {

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					v :=
						x.At(
							n,
							c,
							h,
							w,
						)

					norm :=
						(v - op.Mean.At(c)) /
							float32(
								math.Sqrt(
									float64(
										op.Variance.At(c)+op.Eps,
									),
								),
							)

					sum +=
						grad.At(
							n,
							c,
							h,
							w,
						) *
							norm

				}
			}
		}

		dgamma.Set(
			sum,
			c,
		)

	}
	for n := 0; n < batch; n++ {

		for c := 0; c < channels; c++ {

			sumDy :=
				float32(0)

			sumDyNorm :=
				float32(0)

			for i := 0; i < batch; i++ {

				for h := 0; h < height; h++ {

					for w := 0; w < width; w++ {

						g :=
							grad.At(
								i,
								c,
								h,
								w,
							)

						sumDy += g

						norm :=
							(x.At(
								i,
								c,
								h,
								w,
							) - op.Mean.At(c)) /
								float32(
									math.Sqrt(
										float64(
											op.Variance.At(c)+op.Eps,
										),
									),
								)

						sumDyNorm +=
							g * norm

					}
				}
			}

			for h := 0; h < height; h++ {

				for w := 0; w < width; w++ {

					norm :=
						(x.At(
							n,
							c,
							h,
							w,
						) - op.Mean.At(c)) /
							float32(
								math.Sqrt(
									float64(
										op.Variance.At(c)+op.Eps,
									),
								),
							)

					d :=
						grad.At(
							n,
							c,
							h,
							w,
						)

					value :=
						(gamma.At(c) /
							float32(
								math.Sqrt(
									float64(
										op.Variance.At(c)+op.Eps,
									),
								),
							)) *
							(size*d -
								sumDy -
								norm*sumDyNorm) /
							size

					dx.Set(
						value,
						n,
						c,
						h,
						w,
					)

				}
			}
		}
	}

	return []tensor.Tensor{

		dx,

		dgamma,

		dbeta,
	}, nil
}
