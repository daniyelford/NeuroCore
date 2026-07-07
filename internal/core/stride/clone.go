package stride

func (s Stride) Clone() Stride {
	return Stride{
		vector: s.vector.Clone(),
	}
}
