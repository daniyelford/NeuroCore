package dtype

func (d DType) Size() int {

	switch d {

	case Int8, Uint8, Bool:
		return 1

	case Int16, Uint16:
		return 2

	case Int32, Uint32, Float32:
		return 4

	case Int64, Uint64, Float64:
		return 8

	default:
		return 0

	}

}