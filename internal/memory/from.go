package memory

import "github.com/daniyelford/neurocore/internal/core/backend"

func From(data []float32) Memory {

	m := NewOnDevice(
		len(data),
		backend.CPU,
	)

	copy(m.data, data)

	return m
}
