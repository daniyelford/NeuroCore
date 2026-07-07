package impl

import (
	"sync/atomic"

	"github.com/daniyelford/neurocore/internal/core/device"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/stride"
	"github.com/daniyelford/neurocore/internal/memory/storage"
)

type TensorImpl struct {
	storage *storage.Storage

	shape  *shape.Shape
	stride *stride.Stride

	offset int

	device device.Device

	version atomic.Uint64
}
