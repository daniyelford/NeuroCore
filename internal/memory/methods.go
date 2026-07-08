package memory

func (m Memory) Len() int {

	return len(m.data)

}

func (m Memory) Empty() bool {

	return len(m.data) == 0

}
func (m Memory) Values() []float32 {

	out := make([]float32, len(m.data))

	copy(out, m.data)

	return out

}
