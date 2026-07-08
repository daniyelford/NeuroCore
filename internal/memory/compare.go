package memory

func (m Memory) Equal(other Memory) bool {

	if len(m.data) != len(other.data) {

		return false

	}

	for i := range m.data {

		if m.data[i] != other.data[i] {

			return false

		}

	}

	return true

}
