package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (t Tensor) Flatten() (Tensor, bool) {

	return t.Reshape(
		shape.New(
			t.NumElements(),
		),
	)

}
