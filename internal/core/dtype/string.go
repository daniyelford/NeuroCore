package dtype

func (d DType) String() string {

	switch d {

	case Int8:
		return "int8"

	case Int16:
		return "int16"

	case Int32:
		return "int32"

	case Int64:
		return "int64"

	case Uint8:
		return "uint8"

	case Uint16:
		return "uint16"

	case Uint32:
		return "uint32"

	case Uint64:
		return "uint64"

	case Float32:
		return "float32"

	case Float64:
		return "float64"

	case Bool:
		return "bool"

	default:
		return "invalid"

	}

}