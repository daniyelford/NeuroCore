package ndim

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

// Get returns the value at index i.
// The second return value is false if the index is out of range.
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

// IsScalar reports whether the vector has exactly one dimension.
func (v Vector) IsScalar() bool {
	return len(v.values) == 1
}

// IsVector reports whether the vector has more than one dimension.
func (v Vector) IsVector() bool {
	return len(v.values) > 1
}
