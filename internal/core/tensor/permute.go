package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (t Tensor) Permute(
	dims ...int,
) (Tensor, bool) {

	old := t.Shape().Values()

	if len(dims) != len(old) {
		return Tensor{}, false
	}

	used := make([]bool, len(old))

	newShape := make([]int, len(dims))

	for i, d := range dims {

		if d < 0 || d >= len(old) {
			return Tensor{}, false
		}

		if used[d] {
			return Tensor{}, false
		}

		used[d] = true

		newShape[i] = old[d]

	}

	out :=
		New(
			shape.New(newShape...),
		)

	for i := 0; i < t.Len(); i++ {

		// فعلا copy ساده
		out.FlatSet(
			i,
			t.FlatAt(i),
		)

	}

	return out, true
}
