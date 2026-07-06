package storage

import "github.com/daniyelford/neurocore/internal/memory/buffer"

func (s *Storage) Buffer() buffer.Buffer {

    return s.buffer

}