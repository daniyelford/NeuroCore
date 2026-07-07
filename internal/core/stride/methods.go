package stride

func (s Stride) Rank() int {
	return s.vector.Rank()
}

func (s Stride) Len() int {
	return s.vector.Len()
}

func (s Stride) At(i int) int {
	return s.vector.At(i)
}

func (s Stride) Get(i int) (int, bool) {
	return s.vector.Get(i)
}

func (s Stride) Values() []int {
	return s.vector.Values()
}

// Last returns the last stride.
func (s Stride) Last() int {
	return s.vector.Last()
}
