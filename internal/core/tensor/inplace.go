package tensor

func (t Tensor) AddInplace(
	other Tensor,
) bool {

	if !t.shape.Equal(other.shape) {

		return false

	}

	for i := 0; i < t.Len(); i++ {

		t.memory.Set(
			i,
			t.memory.At(i)+other.memory.At(i),
		)

	}

	return true

}
func (t Tensor) SubInplace(
	other Tensor,
) bool {

	if !t.shape.Equal(other.shape) {

		return false

	}

	for i := 0; i < t.Len(); i++ {

		t.memory.Set(
			i,
			t.memory.At(i)-other.memory.At(i),
		)

	}

	return true

}
func (t Tensor) ScaleInplace(
	value float32,
) {

	for i := 0; i < t.Len(); i++ {

		t.memory.Set(
			i,
			t.memory.At(i)*value,
		)

	}

}
