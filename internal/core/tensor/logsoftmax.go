package tensor

import "math"

func (t Tensor) LogSoftmax() Tensor {

	return t.LogSoftmaxDim(
		0,
	)

}
func (t Tensor) LogSoftmaxDim(
	axis int,
) Tensor {

	dims :=
		t.Shape().Values()

	// فعلاً فقط 2D axis=1
	if len(dims) != 2 || axis != 1 {

		return t.LogSoftmax()

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

		// sum(exp(x-max))
		sum :=
			float32(0)

		for c := 0; c < cols; c++ {

			v :=
				t.At(
					r,
					c,
				)

			sum +=
				float32(
					math.Exp(
						float64(
							v - max,
						),
					),
				)

		}

		logSum :=
			float32(
				math.Log(
					float64(sum),
				),
			)

		// x - max - log(sum(exp(x-max)))
		for c := 0; c < cols; c++ {

			value :=
				t.At(
					r,
					c,
				)

			out.Set(
				value-max-logSum,
				r,
				c,
			)

		}

	}

	return out

}
