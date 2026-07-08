package device

func Default() Device {

	cpu := NewCPU()

	return cpu

}
func init() {

	Register(
		"cpu",
		NewCPU(),
	)

}
