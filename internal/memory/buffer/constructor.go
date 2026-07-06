package buffer

import "fmt"

func NewCPU(elements int, dt dtype.DType) (*CPUBuffer, error) {
	if elements < 0 {
		return nil, fmt.Errorf("elements cannot be negative")
	}

	size := int(dt.Size())
	if size == 0 {
		return nil, fmt.Errorf("invalid dtype")
	}

	return &CPUBuffer{
		data:  make([]byte, elements*size),
		dtype: dt,
	}, nil
}