package ndim

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

// Count returns the number of occurrences of value.
func (v Vector) Count(value int) int {

	count := 0

	for _, x := range v.values {

		if x == value {
			count++
		}

	}

	return count

}
