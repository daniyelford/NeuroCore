package shape

func (s *Shape) IsScalar() bool {

	return s.Rank()==0

}

func (s *Shape) IsVector() bool {

	return s.Rank()==1

}

func (s *Shape) IsMatrix() bool {

	return s.Rank()==2

}