package dtype

import "fmt"

func Parse(s string) (DType, error) {
	switch s {
	case "bool":
		return Bool, nil
	case "int8":
		return Int8, nil
	case "int16":
		return Int16, nil
	case "int32":
		return Int32, nil
	case "int64":
		return Int64, nil
	case "uint8":
		return UInt8, nil
	case "uint16":
		return UInt16, nil
	case "uint32":
		return UInt32, nil
	case "uint64":
		return UInt64, nil
	case "float32":
		return Float32, nil
	case "float64":
		return Float64, nil
	default:
		return Invalid, fmt.Errorf("unknown dtype: %s", s)
	}
}