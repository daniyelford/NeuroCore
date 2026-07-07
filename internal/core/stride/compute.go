package stride

import (
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func Compute(sh shape.Shape, order layout.Order) Stride {

	if !order.Valid() {
		order = layout.Default()
	}

	switch order {

	case layout.RowMajor:
		return computeRowMajor(sh)

	case layout.ColumnMajor:
		return computeColumnMajor(sh)

	default:
		return computeRowMajor(sh)

	}

}

func computeRowMajor(sh shape.Shape) Stride {

	dims := sh.Values()

	if len(dims) == 0 {
		return New()
	}

	out := make([]int, len(dims))

	step := 1

	for i := len(dims) - 1; i >= 0; i-- {

		out[i] = step
		step *= dims[i]

	}

	return New(out...)

}

func computeColumnMajor(sh shape.Shape) Stride {

	dims := sh.Values()

	if len(dims) == 0 {
		return New()
	}

	out := make([]int, len(dims))

	step := 1

	for i := 0; i < len(dims); i++ {

		out[i] = step
		step *= dims[i]

	}

	return New(out...)

}
