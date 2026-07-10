package tensor

func (t Tensor) Scale(
	value float32,
) Tensor {

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)*value,
		)

	}

	return out

}
func (t Tensor) Neg() Tensor {

	return t.Scale(-1)

}
func (t Tensor) AddScalar(
	value float32,
) Tensor {

	out := New(t.shape)

	for i := 0; i < t.Len(); i++ {

		out.memory.Set(
			i,
			t.memory.At(i)+value,
		)

	}

	return out

}
func (t Tensor) Dot(
	other Tensor,
) (float32, bool) {

	if t.Len() != other.Len() {

		return 0, false

	}

	var sum float32

	for i := 0; i < t.Len(); i++ {

		sum +=
			t.memory.At(i) *
				other.memory.At(i)

	}

	return sum, true

}
func (t Tensor) DivScalar(
	v float32,
) Tensor {

	out := t.Clone()

	for i := 0; i < out.NumElements(); i++ {

		idx := out.memoryIndex(i)

		out.memory.Set(
			idx,
			out.memory.At(idx)/v,
		)

	}

	return out

}
