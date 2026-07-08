package tensor

func (t Tensor) Fill(
	value float32,
) {

	t.memory.Fill(value)

}

func (t Tensor) Zero() {

	t.Fill(0)

}
