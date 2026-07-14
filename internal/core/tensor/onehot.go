package tensor

import "github.com/daniyelford/neurocore/internal/core/shape"

func OneHot(
	index int,
	classes int,
) Tensor {

	out :=
		New(
			shape.New(classes),
		)

	out.FlatSet(
		index,
		1,
	)

	return out

}
