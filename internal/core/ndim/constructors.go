package ndim

// New creates a new immutable vector.
func New(values ...int) Vector {
	out := make([]int, len(values))
	copy(out, values)

	return Vector{
		values: out,
	}
}

// Zeros creates a vector of length n filled with zeros.
func Zeros(n int) Vector {
	return Vector{
		values: make([]int, n),
	}
}

// Ones creates a vector of length n filled with ones.
func Ones(n int) Vector {

	out := make([]int, n)

	for i := range out {
		out[i] = 1
	}

	return Vector{
		values: out,
	}
}
