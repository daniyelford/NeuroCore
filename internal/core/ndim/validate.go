package ndim

func (v Vector) Valid() bool {

	for _, x := range v.values {

		if x <= 0 {
			return false
		}

	}

	return true

}
