package tensor

func (t Tensor) MulScalar(
	value float32,
) Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.Len(); i++ {

		out.FlatSet(
			i,
			t.FlatAt(i)*value,
		)

	}

	return out

}
