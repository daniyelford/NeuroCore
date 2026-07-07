package ndim

// Values returns a copy of the underlying values.
func (v Vector) Values() []int {

	out := make([]int, len(v.values))

	copy(out, v.values)

	return out

}
