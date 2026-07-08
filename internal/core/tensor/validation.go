package tensor

// Valid checks internal tensor consistency.
func (t Tensor) Valid() bool {

	if !t.shape.Valid() {
		return false
	}

	if !t.stride.Valid() {
		return false
	}

	return t.shape.NumElements() == t.memory.Len()

}
