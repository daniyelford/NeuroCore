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