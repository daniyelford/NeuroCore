package memory

import "github.com/daniyelford/neurocore/internal/core/backend"

func (m Memory) Zero() {

	for i := range m.data {

		m.data[i] = 0

	}

}
func (m Memory) Valid() bool {

	return m.data != nil

}
func (m Memory) Device() backend.DeviceType {

	return m.device

}
