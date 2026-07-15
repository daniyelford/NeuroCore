package ndim

import "fmt"

func (v Vector) Rank() int {
	return len(v.values)
}
func (v Vector) Len() int {
	return len(v.values)
}
func (v Vector) Empty() bool {
	return len(v.values) == 0
}
func (v Vector) At(i int) int {
	return v.values[i]
}
func (v Vector) Get(i int) (int, bool) {
	if i < 0 || i >= len(v.values) {
		return 0, false
	}
	return v.values[i], true
}
func (v Vector) First() int {
	return v.values[0]
}
func (v Vector) Last() int {
	return v.values[len(v.values)-1]
}
func (v Vector) IsScalar() bool {
	return len(v.values) == 1
}
func (v Vector) IsVector() bool {
	return len(v.values) > 1
}
func (v Vector) Values() []int {
	out := make([]int, len(v.values))
	copy(out, v.values)
	return out
}
func (v Vector) Valid() bool {
	for _, x := range v.values {
		if x <= 0 {
			return false
		}
	}
	return true
}
func (v Vector) String() string {
	return fmt.Sprint(v.values)
}
func (v Vector) Contains(x int) bool {
	for _, value := range v.values {
		if value == x {
			return true
		}
	}
	return false
}
func (v Vector) IndexOf(x int) int {
	for i, value := range v.values {
		if value == x {
			return i
		}
	}
	return -1
}
func (v Vector) Count(value int) int {
	count := 0
	for _, x := range v.values {
		if x == value {
			count++
		}
	}
	return count
}
func (v Vector) With(index int, value int) Vector {
	out := v.Values()
	out[index] = value
	return New(out...)
}
func (v Vector) Append(values ...int) Vector {
	out := v.Values()
	out = append(out, values...)
	return New(out...)
}
func (v Vector) Prepend(values ...int) Vector {
	out := append(values, v.values...)
	return New(out...)
}
func (v Vector) Remove(index int) Vector {
	out := make([]int, 0, len(v.values)-1)
	out = append(out, v.values[:index]...)
	out = append(out, v.values[index+1:]...)
	return New(out...)
}
func (v Vector) Slice(begin, end int) Vector {
	return New(v.values[begin:end]...)
}
func (v Vector) Reverse() Vector {
	out := v.Values()
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return New(out...)
}
func (v Vector) Sum() int {
	sum := 0
	for _, x := range v.values {
		sum += x
	}
	return sum
}
func (v Vector) Product() int {
	if len(v.values) == 0 {
		return 0
	}
	product := 1
	for _, x := range v.values {
		product *= x
	}
	return product
}
func (v Vector) Max() int {
	if len(v.values) == 0 {
		return 0
	}
	max := v.values[0]
	for _, x := range v.values {
		if x > max {
			max = x
		}
	}
	return max
}
func (v Vector) Min() int {
	if len(v.values) == 0 {
		return 0
	}
	min := v.values[0]
	for _, x := range v.values {
		if x < min {
			min = x
		}
	}
	return min

}
func (v Vector) Range(fn func(index, value int) bool) {
	for i, value := range v.values {
		if !fn(i, value) {
			return
		}
	}
}
func (v Vector) Equal(other Vector) bool {
	if len(v.values) != len(other.values) {
		return false
	}
	for i := range v.values {
		if v.values[i] != other.values[i] {
			return false
		}
	}
	return true
}
func (v Vector) Clone() Vector {
	out := make([]int, len(v.values))
	copy(out, v.values)
	return Vector{values: out}
}
