package storage

import (
    "sync/atomic"

    "github.com/daniyelford/neurocore/internal/memory/buffer"
)

type Storage struct {

    buffer buffer.Buffer

    refs atomic.Int64
}