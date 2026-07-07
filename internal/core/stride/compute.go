package stride

import (
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

// Compute computes strides for the requested memory layout.
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
