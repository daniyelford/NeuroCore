package stride

func New(values ...int) (*Stride, error) {

	if len(values) == 0 {
		return nil, ErrEmptyStride
	}

	cp := make([]int, len(values))

	for i, v := range values {

		if v <= 0 {
			return nil, ErrInvalidStride
		}

		cp[i] = v
	}

	return &Stride{
		values: cp,
	}, nil
}