package dtype

func ByName(name string) (DType, bool) {

	d, ok := registry[name]

	return d, ok

}
func (d DType) IsBool() bool {
	return d.kind == KindBool
}

func (d DType) IsInteger() bool {
	return d.kind == KindSignedInteger ||
		d.kind == KindUnsignedInteger
}

func (d DType) IsSigned() bool {
	return d.kind == KindSignedInteger
}

func (d DType) IsUnsigned() bool {
	return d.kind == KindUnsignedInteger
}

func (d DType) IsFloat() bool {
	return d.kind == KindFloatingPoint
}

func (d DType) IsComplex() bool {
	return d.kind == KindComplex
}

func (d DType) IsNumeric() bool {
	return d.IsInteger() ||
		d.IsFloat() ||
		d.IsComplex()
}
func (d DType) Size() uint8 {
	return d.size
}

func (d DType) Kind() Kind {
	return d.kind
}

func (d DType) Name() string {
	return d.name
}
func (d DType) String() string {
	return d.name
}
func (d DType) Valid() bool {
	return d.kind != KindUnknown
}
func (d DType) Equal(other DType) bool {
	return d == other
}
