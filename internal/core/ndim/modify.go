package ndim

// With returns a copy of the vector with one value replaced.
func (v Vector) With(index int, value int) Vector {

	out := v.Values()

	out[index] = value

	return New(out...)

}

// Append returns a new vector with appended values.
func (v Vector) Append(values ...int) Vector {

	out := v.Values()

	out = append(out, values...)

	return New(out...)

}

// Prepend returns a new vector with prepended values.
func (v Vector) Prepend(values ...int) Vector {

	out := append(values, v.values...)

	return New(out...)

}

// Remove removes one element.
func (v Vector) Remove(index int) Vector {

	out := make([]int, 0, len(v.values)-1)

	out = append(out, v.values[:index]...)
	out = append(out, v.values[index+1:]...)

	return New(out...)

}

// Slice returns a sub vector.
func (v Vector) Slice(begin, end int) Vector {

	return New(v.values[begin:end]...)

}

// Reverse returns a reversed copy.
func (v Vector) Reverse() Vector {

	out := v.Values()

	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {

		out[i], out[j] = out[j], out[i]

	}

	return New(out...)

}
