package device

type CPUDevice struct{}

func (CPUDevice) Type() Type {

	return CPU

}

func (CPUDevice) Index() int {

	return 0

}

func (CPUDevice) Name() string {

	return "CPU"

}

func (CPUDevice) IsAvailable() bool {

	return true

}

func (CPUDevice) String() string {

	return "cpu"

}