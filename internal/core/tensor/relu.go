package tensor

func (t Tensor) ReLU() Tensor {

	out := t.Clone()

	for i := 0; i < out.NumElements(); i++ {

		idx := out.memoryIndex(i)

		v := out.memory.At(idx)

		if v < 0 {

			out.memory.Set(idx, 0)

		}

	}

	return out

}
func (t Tensor) ReLUMask() Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		v := t.memory.At(idx)

		if v > 0 {

			out.memory.Set(
				out.memoryIndex(i),
				1,
			)

		} else {

			out.memory.Set(
				out.memoryIndex(i),
				0,
			)

		}

	}

	return out
}
