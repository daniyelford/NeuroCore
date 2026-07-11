package tensor

import (
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func Stack(
	tensors []Tensor,
) (Tensor, bool) {
	if len(tensors) == 0 {

		return Tensor{}, false

	}

	baseShape :=
		tensors[0].Shape().Values()

	resultShape :=
		append(
			[]int{len(tensors)},
			baseShape...,
		)

	out :=
		New(
			shape.New(
				resultShape...,
			),
		)

	offset := 0

	for _, t := range tensors {

		if !t.Shape().Equal(
			tensors[0].Shape(),
		) {

			return Tensor{}, false

		}
		for i := 0; i < t.Len(); i++ {

			out.FlatSet(
				offset+i,
				t.FlatAt(i),
			)

		}

		offset += t.Len()

	}

	return out, true

}
