package tensor

import "github.com/daniyelford/neurocore/internal/core/stride"

// Squeeze removes all dimensions whose size is 1.
// It returns a view and does not copy memory.
func (t Tensor) Squeeze() Tensor {

	out := t

	out.shape = t.shape.Squeeze()

	out.stride = stride.FromShape(out.shape)

	return out
}
