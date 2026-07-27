/*
Package shape defines tensor dimensions.

Shape is immutable and internally uses ndim.Vector.
*/
package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

// Shape represents tensor dimensions.
type Shape struct {
	vector ndim.Vector
}
