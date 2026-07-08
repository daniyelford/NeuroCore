package memory

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
)

func New(size int) Memory {

	return NewOnDevice(
		size,
		backend.CPU,
	)

}

func NewOnDevice(
	size int,
	dev backend.DeviceType,
) Memory {

	if size < 0 {

		size = 0

	}

	return Memory{

		data: make([]float32, size),

		capacity: size,

		device: dev,
	}

}
func NewView(
	m *Memory,
	offset int,
	length int,
) View {

	return View{

		memory: m,

		offset: offset,

		length: length,
	}

}
