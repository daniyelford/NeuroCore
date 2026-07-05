package shape

import "fmt"

func New(dims ...int) (*Shape, error) {

	if len(dims) == 0 {
		return nil, fmt.Errorf("shape cannot be empty")
	}

	for _, d := range dims {
		if d <= 0 {
			return nil, fmt.Errorf("invalid dimension: %d", d)
		}
	}

	cp := make([]int, len(dims))
	copy(cp, dims)

	return &Shape{
		dims: cp,
	}, nil
}