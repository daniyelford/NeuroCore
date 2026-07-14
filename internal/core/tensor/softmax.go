package tensor

import "math"

func (t Tensor) Softmax() Tensor {

	return t.SoftmaxDim(
		0,
	)

}
func (t Tensor) SoftmaxDim(
	axis int,
) Tensor {

	dims :=
		t.Shape().Values()

	if len(dims) != 2 || axis != 1 {

		return t.Softmax()

	}

	rows :=
		dims[0]

	cols :=
		dims[1]

	out :=
		New(
			t.Shape(),
		)

	for r := 0; r < rows; r++ {

		// max row
		max :=
			t.At(
				r,
				0,
			)

		for c := 1; c < cols; c++ {

			v :=
				t.At(
					r,
					c,
				)

			if v > max {

				max = v

			}

		}

		// exp(x-max)
		sum :=
			float32(0)

		for c := 0; c < cols; c++ {

			v :=
				t.At(
					r,
					c,
				)

			e :=
				float32(
					math.Exp(
						float64(v - max),
					),
				)

			out.Set(
				e,
				r,
				c,
			)

			sum += e

		}

		// normalize
		for c := 0; c < cols; c++ {

			v :=
				out.At(
					r,
					c,
				)

			out.Set(
				v/sum,
				r,
				c,
			)

		}

	}

	return out

}
