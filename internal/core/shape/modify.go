package shape

// Expand inserts a dimension of size 1 at index.
func (s Shape) Expand(index int) Shape {

	values := s.Values()

	if index < 0 {
		index = 0
	}

	if index > len(values) {
		index = len(values)
	}

	out := make([]int, 0, len(values)+1)

	out = append(out, values[:index]...)
	out = append(out, 1)
	out = append(out, values[index:]...)

	return New(out...)

}

// Squeeze removes every dimension equal to one.
func (s Shape) Squeeze() Shape {

	values := s.Values()

	out := make([]int, 0, len(values))

	for _, v := range values {

		if v != 1 {
			out = append(out, v)
		}

	}

	if len(out) == 0 {
		out = append(out, 1)
	}

	return New(out...)

}

// Append appends dimensions.
func (s Shape) Append(dim ...int) Shape {

	values := append(s.Values(), dim...)

	return New(values...)

}

// Prepend prepends dimensions.
func (s Shape) Prepend(dim ...int) Shape {

	values := append(dim, s.Values()...)

	return New(values...)

}
