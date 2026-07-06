package storage

import "github.com/daniyelford/neurocore/internal/memory/buffer"

func New(buf buffer.Buffer) *Storage {

    s := &Storage{
        buffer: buf,
    }

    s.refs.Store(1)

    return s
}