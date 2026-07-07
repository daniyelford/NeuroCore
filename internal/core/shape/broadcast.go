package shape

// CanBroadcast reports whether two shapes are broadcast compatible.
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

// BroadcastShape returns the resulting shape after broadcasting.
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
