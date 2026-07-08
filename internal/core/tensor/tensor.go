package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
	"github.com/daniyelford/neurocore/internal/memory"
)

type Tensor struct {
	shape shape.Shape

	stride stride.Stride

	memory memory.Memory

	offset int

	device backend.DeviceType

	layout layout.Order
}
