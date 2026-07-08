package device

import (
	"github.com/daniyelford/neurocore/internal/memory"
)

type CPUDevice struct{}

func NewCPU() CPUDevice {

	return CPUDevice{}

}

func (CPUDevice) Type() Type {

	return CPU

}

func (CPUDevice) Name() string {

	return "cpu"

}

func (CPUDevice) Allocate(size int) memory.Memory {

	return memory.New(size)

}

func (CPUDevice) Copy(
	dst memory.Memory,
	src memory.Memory,
) {

	memory.Copy(dst, src)

}
