package tensor

func (t Tensor) Add(
	other Tensor,
) Tensor {

	if !t.shape.Equal(other.shape) {

		panic("shape mismatch")

	}

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)+other.memory.At(i),
		)

	}

	return out

}
func (t Tensor) Sub(
	other Tensor,
) Tensor {

	if !t.shape.Equal(other.shape) {

		panic("shape mismatch")

	}

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)-other.memory.At(i),
		)

	}

	return out

}
func (t Tensor) Mul(
	other Tensor,
) Tensor {

	if !t.shape.Equal(other.shape) {

		panic("shape mismatch")

	}

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)*other.memory.At(i),
		)

	}

	return out

}
func (t Tensor) Div(
	other Tensor,
) Tensor {

	if !t.shape.Equal(other.shape) {

		panic("shape mismatch")

	}

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)/other.memory.At(i),
		)

	}

	return out

}
func (t Tensor) AddBroadcast(
	other Tensor,
) (Tensor, bool) {

	sh, ok := broadcastShape(
		t.shape,
		other.shape,
	)

	if !ok {

		return Tensor{}, false

	}

	out := New(sh)

	for i := 0; i < out.Len(); i++ {

		av := t.broadcastAt(i, sh)

		bv := other.broadcastAt(i, sh)

		out.memory.Set(
			i,
			av+bv,
		)

	}

	return out, true

}
