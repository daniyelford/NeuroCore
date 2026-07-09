package memory

func (m Memory) Zero() {

	for i := range m.data {

		m.data[i] = 0

	}

}
func (m Memory) Valid() bool {

	return m.data != nil

}
