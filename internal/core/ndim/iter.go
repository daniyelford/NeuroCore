package ndim

// Range calls fn for every value.
// If fn returns false, iteration stops.
// Range calls fn for every value until fn returns false.
func (v Vector) Range(fn func(index, value int) bool) {
	for i, value := range v.values {
		if !fn(i, value) {
			return
		}
	}
}
