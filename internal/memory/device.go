package memory

import "github.com/daniyelford/neurocore/internal/core/backend"

func (m Memory) Device() backend.DeviceType {
	return m.device
}
