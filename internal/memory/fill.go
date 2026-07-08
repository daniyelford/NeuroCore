package memory

func (m Memory) Fill(value float32) {

	for i := range m.data {

		m.data[i] = value

	}

}
