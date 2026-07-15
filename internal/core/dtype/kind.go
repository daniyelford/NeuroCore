package dtype

type Kind uint8

const (
	KindUnknown Kind = iota
	KindBool
	KindSignedInteger
	KindUnsignedInteger
	KindFloatingPoint
	KindComplex
)

var (
	Bool = DType{
		name: "bool",
		size: 1,
		kind: KindBool,
	}

	Int8 = DType{
		name: "int8",
		size: 1,
		kind: KindSignedInteger,
	}

	Int16 = DType{
		name: "int16",
		size: 2,
		kind: KindSignedInteger,
	}

	Int32 = DType{
		name: "int32",
		size: 4,
		kind: KindSignedInteger,
	}

	Int64 = DType{
		name: "int64",
		size: 8,
		kind: KindSignedInteger,
	}

	Uint8 = DType{
		name: "uint8",
		size: 1,
		kind: KindUnsignedInteger,
	}

	Uint16 = DType{
		name: "uint16",
		size: 2,
		kind: KindUnsignedInteger,
	}

	Uint32 = DType{
		name: "uint32",
		size: 4,
		kind: KindUnsignedInteger,
	}

	Uint64 = DType{
		name: "uint64",
		size: 8,
		kind: KindUnsignedInteger,
	}

	Float32 = DType{
		name: "float32",
		size: 4,
		kind: KindFloatingPoint,
	}

	Float64 = DType{
		name: "float64",
		size: 8,
		kind: KindFloatingPoint,
	}

	Complex64 = DType{
		name: "complex64",
		size: 8,
		kind: KindComplex,
	}

	Complex128 = DType{
		name: "complex128",
		size: 16,
		kind: KindComplex,
	}
)
var registry = map[string]DType{

	"bool": Bool,

	"int8": Int8,

	"int16": Int16,

	"int32": Int32,

	"int64": Int64,

	"uint8": Uint8,

	"uint16": Uint16,

	"uint32": Uint32,

	"uint64": Uint64,

	"float32": Float32,

	"float64": Float64,

	"complex64": Complex64,

	"complex128": Complex128,
}
