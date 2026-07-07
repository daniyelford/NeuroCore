package stride

// Offset computes the linear memory offset.
//
// This method assumes:
//
//  • number of indices == Rank()
//  • every index is valid
//
// For a safe version use TryOffset().
func (s Stride) Offset(indices ...int) int {

	offset := 0

	for i, index := range indices {

		offset += index * s.At(i)

	}

	return offset

}

// TryOffset safely computes the memory offset.
func (s Stride) TryOffset(indices ...int) (int, bool) {

	if len(indices) != s.Len() {
		return 0, false
	}

	offset := 0

	for i, index := range indices {

		if index < 0 {
			return 0, false
		}

		offset += index * s.At(i)

	}

	return offset, true

}
