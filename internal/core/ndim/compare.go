package ndim

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
