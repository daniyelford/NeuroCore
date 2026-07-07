package dtype

func (d DType) Valid() bool {
	return d.kind != KindUnknown
}