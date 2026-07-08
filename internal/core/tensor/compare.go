package tensor

func (t Tensor) Equal(
	other Tensor,
) bool {

	return t.shape.Equal(other.shape) &&
		t.memory.Equal(other.memory)

}
