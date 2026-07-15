package shape

import "github.com/daniyelford/neurocore/internal/core/ndim"

func New(dimensions ...int) Shape {
	return Shape{
		vector: ndim.New(dimensions...),
	}
}
func (s Shape) Rank() int {
	return s.vector.Rank()
}
func (s Shape) Len() int {
	return s.vector.Len()
}
func (s Shape) At(i int) int {
	return s.vector.At(i)
}
func (s Shape) Get(i int) (int, bool) {
	return s.vector.Get(i)
}
func (s Shape) First() int {
	return s.vector.First()
}
func (s Shape) Last() int {
	return s.vector.Last()
}
func (s Shape) Values() []int {
	return s.vector.Values()
}
func (s Shape) cloneVector() ndim.Vector {
	return s.vector.Clone()
}
func (s Shape) Valid() bool {
	return s.vector.Valid()
}
func (s Shape) String() string {
	return s.vector.String()
}
func (s Shape) Dimensions() []int {
	return s.Values()
}
func (s Shape) Contains(dim int) bool {
	return s.vector.Contains(dim)
}
func (s Shape) IndexOf(dim int) int {
	return s.vector.IndexOf(dim)
}
func (s Shape) CanReshape(target Shape) bool {
	return s.NumElements() == target.NumElements()
}
func (s Shape) IsScalar() bool {
	return s.Len() == 1 && s.At(0) == 1
}
func (s Shape) IsVector() bool {
	return s.Len() == 1 && s.At(0) > 1
}
func (s Shape) IsMatrix() bool {
	return s.Len() == 2
}
func (s Shape) IsTensor() bool {
	return s.Len() >= 3
}
func (s Shape) Expand(index int) Shape {
	values := s.Values()
	if index < 0 {
		index = 0
	}
	if index > len(values) {
		index = len(values)
	}
	out := make([]int, 0, len(values)+1)
	out = append(out, values[:index]...)
	out = append(out, 1)
	out = append(out, values[index:]...)
	return New(out...)
}
func (s Shape) Squeeze() Shape {
	values := s.Values()
	out := make([]int, 0, len(values))
	for _, v := range values {
		if v != 1 {
			out = append(out, v)
		}
	}
	if len(out) == 0 {
		out = append(out, 1)
	}
	return New(out...)
}
func (s Shape) Append(dim ...int) Shape {
	values := append(s.Values(), dim...)
	return New(values...)
}
func (s Shape) Prepend(dim ...int) Shape {
	values := append(dim, s.Values()...)
	return New(values...)
}
func (s Shape) NumElements() int {
	if s.Len() == 0 {
		return 0
	}
	product := 1
	s.vector.Range(func(_ int, value int) bool {
		product *= value
		return true
	})
	return product
}
func newFromVector(v ndim.Vector) Shape {
	return Shape{
		vector: v,
	}
}
func (s Shape) Equal(other Shape) bool {
	return s.vector.Equal(other.vector)
}
func (s Shape) Clone() Shape {
	return Shape{
		vector: s.vector.Clone(),
	}
}
func (s Shape) CanBroadcast(other Shape) bool {
	a := s.Values()
	b := other.Values()
	i := len(a) - 1
	j := len(b) - 1
	for i >= 0 && j >= 0 {
		if a[i] != b[j] &&
			a[i] != 1 &&
			b[j] != 1 {
			return false
		}
		i--
		j--
	}
	return true
}
func (s Shape) BroadcastShape(other Shape) (Shape, bool) {
	if !s.CanBroadcast(other) {
		return Shape{}, false
	}
	a := s.Values()
	b := other.Values()
	size := len(a)
	if len(b) > size {
		size = len(b)
	}
	result := make([]int, size)
	i := len(a) - 1
	j := len(b) - 1
	k := size - 1
	for k >= 0 {
		av := 1
		bv := 1
		if i >= 0 {
			av = a[i]
		}
		if j >= 0 {
			bv = b[j]
		}
		if av > bv {
			result[k] = av
		} else {
			result[k] = bv
		}
		i--
		j--
		k--
	}
	return New(result...), true
}
