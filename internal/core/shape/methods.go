package shape

func (s *Shape) Rank() int {
	return len(s.dims)
}

func (s *Shape) Dims() []int {
	cp := make([]int, len(s.dims))
	copy(cp, s.dims)
	return cp
}

func (s *Shape) Dim(i int) int {
	return s.dims[i]
}

func (s *Shape) NumElements() int {

	total := 1

	for _, d := range s.dims {
		total *= d
	}

	return total
}