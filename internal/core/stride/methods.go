package stride

import (
	"github.com/daniyelford/neurocore/internal/core/layout"
	"github.com/daniyelford/neurocore/internal/core/shape"
)

func (s Stride) Rank() int {
	return s.vector.Rank()
}
func (s Stride) Len() int {
	return s.vector.Len()
}
func (s Stride) At(i int) int {
	return s.vector.At(i)
}
func (s Stride) Get(i int) (int, bool) {
	return s.vector.Get(i)
}
func (s Stride) Values() []int {
	return s.vector.Values()
}
func (s Stride) Last() int {
	return s.vector.Last()
}
func (s Stride) Valid() bool {
	return s.vector.Valid()
}
func (s Stride) String() string {
	return s.vector.String()
}
func (s Stride) Offset(indices ...int) int {
	offset := 0
	for i, index := range indices {
		offset += index * s.At(i)
	}
	return offset
}
func (s Stride) TryOffset(indices ...int) (int, bool) {
	if len(indices) != s.Len() {
		return 0, false
	}
	offset := 0
	for i, index := range indices {
		if index < 0 {
			return 0, false
		}
		offset += index * s.At(i)
	}
	return offset, true
}
func FromShape(s shape.Shape) Stride {
	dims := s.Values()
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
func (s Stride) Equal(other Stride) bool {
	return s.vector.Equal(other.vector)
}
func (s Stride) Clone() Stride {
	return Stride{vector: s.vector.Clone()}
}
func (s Stride) IsEmpty() bool {
	return s.Len() == 0
}
func (s Stride) IsScalar() bool {
	return s.Len() == 1 && s.At(0) == 1
}
