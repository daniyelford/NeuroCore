package tensor

func binaryOp(
	a Tensor,
	b Tensor,
	op func(
		float32,
		float32,
	) float32,
) Tensor {

	out, ok :=
		broadcastBinary(
			a,
			b,
			op,
		)

	if !ok {

		panic(
			"shape mismatch",
		)

	}

	return out

}

func (t Tensor) Add(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a + b

		},
	)

}

func (t Tensor) Sub(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a - b

		},
	)

}

func (t Tensor) Mul(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a * b

		},
	)

}

func (t Tensor) Div(
	other Tensor,
) Tensor {

	return binaryOp(
		t,
		other,

		func(
			a,
			b float32,
		) float32 {

			return a / b

		},
	)

}
