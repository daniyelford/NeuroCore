package memory

// Copy copies data from source to destination.
func Copy(
	dst Memory,
	src Memory,
) {

	copy(
		dst.data,
		src.data,
	)

}
