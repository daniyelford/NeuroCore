package memory

// Resize changes memory size.
func (m *Memory) Resize(size int) {

	if size < 0 {

		size = 0

	}

	if size <= cap(m.data) {

		m.data = m.data[:size]

		return

	}

	newData := make([]float32, size)

	copy(
		newData,
		m.data,
	)

	m.data = newData

	m.capacity = size

}
