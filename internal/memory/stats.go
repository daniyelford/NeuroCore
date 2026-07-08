package memory

func (m Memory) Capacity() int {

	return m.capacity

}

func (m Memory) Bytes() int {

	return len(m.data) * 4

}
