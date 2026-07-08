package memory

func (m Memory) Clone() Memory {

	data := make([]float32, len(m.data))

	copy(data, m.data)

	return Memory{

		data: data,
	}

}
