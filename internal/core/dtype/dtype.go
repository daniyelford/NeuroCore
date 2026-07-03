package dtype

type DType uint8

const (
	Invalid DType = iota

	Int8
	Int16
	Int32
	Int64

	Uint8
	Uint16
	Uint32
	Uint64

	Float32
	Float64

	Bool
)