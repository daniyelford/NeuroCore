package memory

import "github.com/daniyelford/neurocore/internal/core/device"

type Memory struct {
	data []float32

	capacity int

	device device.Type
}
