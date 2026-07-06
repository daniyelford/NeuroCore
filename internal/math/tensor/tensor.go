package tensor

import (
    "github.com/daniyelford/neurocore/internal/core/shape"
    "github.com/daniyelford/neurocore/internal/memory/storage"
)

type Tensor struct {
    storage *storage.Storage
    shape   shape.Shape
    offset  int
    strides []int
}