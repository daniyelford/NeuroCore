package tensor

func SameShape(
	a Tensor,
	b Tensor,
) bool {

	return a.Shape().Equal(
		b.Shape(),
	)

}

func SameSize(
	a Tensor,
	b Tensor,
) bool {

	return a.Len() == b.Len()

}
