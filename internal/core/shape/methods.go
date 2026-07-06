func (s *Shape) Rank() int{

	return len(s.dims)

}

func (s *Shape) Size() int{

	return s.size

}

func (s *Shape) Dims() []int{

	cp:=make([]int,len(s.dims))

	copy(cp,s.dims)

	return cp

}

func (s *Shape) Dim(i int) int{

	return s.dims[i]

}