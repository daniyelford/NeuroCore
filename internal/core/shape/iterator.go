package shape

func (s *Shape) LastDim() int{

	return s.dims[len(s.dims)-1]

}