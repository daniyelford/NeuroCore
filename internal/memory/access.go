package memory

func (m Memory) At(index int) float32 {

	return m.data[index]

}

func (m Memory) Set(index int, value float32) {

	m.data[index] = value

}
func (m Memory) Get(index int) (float32, bool) {

	if index < 0 || index >= len(m.data) {
		return 0, false
	}

	return m.data[index], true
}
func (m Memory) TrySet(
	index int,
	value float32,
) bool {

	if index < 0 ||
		index >= len(m.data) {

		return false

	}

	m.data[index] = value

	return true

}
func (m Memory) TryAt(index int) (float32, bool) {

	if index < 0 ||
		index >= len(m.data) {

		return 0, false
	}

	return m.data[index], true

}

// CopyFrom copies values into memory.
func (m Memory) CopyFrom(values []float32) {

	copy(
		m.data,
		values,
	)

}
