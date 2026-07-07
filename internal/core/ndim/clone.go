package ndim

func (v Vector) Clone() Vector {

	out := make([]int, len(v.values))

	copy(out, v.values)

	return Vector{
		values: out,
	}

}
