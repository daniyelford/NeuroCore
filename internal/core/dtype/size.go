package dtype

func (d DType) Size() uint8 {
	return d.size
}

func (d DType) Kind() Kind {
	return d.kind
}

func (d DType) Name() string {
	return d.name
}