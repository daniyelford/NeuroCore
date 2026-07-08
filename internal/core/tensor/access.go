package tensor

func (t Tensor) At(
	indices ...int,
) float32 {

	offset := t.stride.Offset(indices...)

	return t.memory.At(offset)

}

func (t Tensor) Set(
	value float32,
	indices ...int,
) {

	offset := t.stride.Offset(indices...)

	t.memory.Set(
		offset,
		value,
	)

}
func (t Tensor) TryAt(
	indices ...int,
) (float32, bool) {

	offset, ok := t.stride.TryOffset(indices...)

	if !ok {
		return 0, false
	}

	return t.memory.TryAt(offset)

}
func (t Tensor) TrySet(
	value float32,
	indices ...int,
) bool {

	offset, ok := t.stride.TryOffset(indices...)

	if !ok {
		return false
	}

	return t.memory.TrySet(
		offset,
		value,
	)

}
