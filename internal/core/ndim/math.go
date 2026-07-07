package ndim

func (v Vector) Sum() int {

	sum := 0

	for _, x := range v.values {
		sum += x
	}

	return sum

}

func (v Vector) Product() int {

	if len(v.values) == 0 {
		return 0
	}

	product := 1

	for _, x := range v.values {
		product *= x
	}

	return product

}

func (v Vector) Max() int {

	if len(v.values) == 0 {
		return 0
	}

	max := v.values[0]

	for _, x := range v.values {

		if x > max {
			max = x
		}

	}

	return max

}

func (v Vector) Min() int {

	if len(v.values) == 0 {
		return 0
	}

	min := v.values[0]

	for _, x := range v.values {

		if x < min {
			min = x
		}

	}

	return min

}
