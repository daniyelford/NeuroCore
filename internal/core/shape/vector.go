package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

// cloneVector returns a deep copy of the underlying vector.
//
// Internal use only.
func (s Shape) cloneVector() ndim.Vector {
	return s.vector.Clone()
}
