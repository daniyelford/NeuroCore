package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

// vector returns the underlying immutable vector.
//
// Internal use only.
func (s Shape) cloneVector() ndim.Vector {
	return s.vector.Clone()
}
