package dtype

func ByName(name string) (DType, bool) {

	d, ok := registry[name]

	return d, ok

}