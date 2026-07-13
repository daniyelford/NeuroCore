package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (t Tensor) Pad(
	top int,
	bottom int,
	left int,
	right int,
	value float32,
) (Tensor, bool) {

	dims :=
		t.Shape().Values()

	if len(dims) != 2 {
		return Tensor{}, false
	}

	h := dims[0]
	w := dims[1]

	out :=
		New(
			shape.New(
				h+top+bottom,
				w+left+right,
			),
		)

	out.Fill(value)

	for i := 0; i < h; i++ {

		for j := 0; j < w; j++ {

			out.Set(
				t.At(i, j),
				i+top,
				j+left,
			)

		}

	}

	return out, true
}
