package memory

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
)

type Memory struct {
	data []float32

	capacity int

	device backend.DeviceType
}
