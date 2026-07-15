package memory

import "github.com/daniyelford/neurocore/internal/core/backend"

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
func (m Memory) Clone() Memory {

	data := make([]float32, len(m.data))

	copy(data, m.data)

	return Memory{

		data: data,
	}

}
func (m Memory) CopyFrom(values []float32) {
	copy(
		m.data,
		values,
	)
}
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
func (m Memory) Device() backend.DeviceType {
	return m.device
}
func (m Memory) Fill(value float32) {

	for i := range m.data {

		m.data[i] = value

	}

}
func (m Memory) Zero() {

	for i := range m.data {

		m.data[i] = 0

	}

}
func (m Memory) Valid() bool {

	return m.data != nil

}
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
func (m Memory) Capacity() int {

	return m.capacity

}

func (m Memory) Bytes() int {

	return len(m.data) * 4

}

func Copy(dst Memory, src Memory) {
	copy(dst.data, src.data)
}
func From(data []float32) Memory {

	m := NewOnDevice(
		len(data),
		backend.CPU,
	)

	copy(m.data, data)

	return m
}
