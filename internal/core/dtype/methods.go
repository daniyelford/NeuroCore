package dtype

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