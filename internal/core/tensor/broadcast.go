package tensor

import "github.com/daniyelford/neurocore/internal/core/shape"

func broadcastShape(
	a shape.Shape,
	b shape.Shape,
) (shape.Shape, bool) {

	ad := a.Values()

	bd := b.Values()

	max := len(ad)

	if len(bd) > max {

		max = len(bd)

	}

	result := make([]int, max)

	for i := 0; i < max; i++ {

		av := 1

		bv := 1

		if len(ad)-1-i >= 0 {

			av = ad[len(ad)-1-i]

		}

		if len(bd)-1-i >= 0 {

			bv = bd[len(bd)-1-i]

		}

		if av != bv &&
			av != 1 &&
			bv != 1 {

			return shape.Shape{}, false

		}

		if av > bv {

			result[max-1-i] = av

		} else {

			result[max-1-i] = bv

		}

	}

	return shape.New(result...), true

}
func (t Tensor) broadcastAt(
	index int,
	target shape.Shape,
) float32 {

	targetIndices :=
		linearToIndices(
			index,
			target,
		)

	srcIndices :=
		broadcastIndices(
			targetIndices,
			t.Shape(),
		)

	return t.At(
		srcIndices...,
	)

}
func (t Tensor) Broadcast(
	target shape.Shape,
) (Tensor, bool) {

	_, ok :=
		broadcastShape(
			t.Shape(),
			target,
		)

	if !ok {

		return Tensor{}, false

	}

	out :=
		New(
			target,
		)

	for i := 0; i < out.NumElements(); i++ {

		out.FlatSet(

			i,

			t.broadcastAt(
				i,
				target,
			),
		)

	}

	return out,
		true

}
func (t Tensor) AddBroadcast(
	other Tensor,
) (
	Tensor,
	bool,
) {

	return broadcastBinary(

		t,

		other,

		func(
			a,
			b float32,
		) float32 {

			return a + b

		},
	)

}
func linearToIndices(
	index int,
	sh shape.Shape,
) []int {

	dims := sh.Values()

	indices :=
		make(
			[]int,
			len(dims),
		)

	for i := len(dims) - 1; i >= 0; i-- {

		indices[i] =
			index % dims[i]

		index /= dims[i]

	}

	return indices

}
func indicesToLinear(
	indices []int,
	sh shape.Shape,
) int {

	dims := sh.Values()

	stride := 1

	index := 0

	for i := len(dims) - 1; i >= 0; i-- {

		index +=
			indices[i] * stride

		stride *= dims[i]

	}

	return index

}
func broadcastIndices(
	targetIndices []int,
	src shape.Shape,
) []int {

	srcDims := src.Values()

	out :=
		make(
			[]int,
			len(srcDims),
		)

	offset :=
		len(targetIndices) -
			len(srcDims)

	for i := range srcDims {

		j := i + offset

		if j < 0 {

			out[i] = 0

			continue

		}

		if srcDims[i] == 1 {

			out[i] = 0

		} else {

			out[i] = targetIndices[j]

		}

	}

	return out

}
func (t Tensor) SubBroadcast(
	other Tensor,
) (
	Tensor,
	bool,
) {

	return broadcastBinary(

		t,

		other,

		func(
			a,
			b float32,
		) float32 {

			return a - b

		},
	)

}
func (t Tensor) MulBroadcast(
	other Tensor,
) (
	Tensor,
	bool,
) {

	return broadcastBinary(

		t,

		other,

		func(
			a,
			b float32,
		) float32 {

			return a * b

		},
	)

}
func (t Tensor) DivBroadcast(
	other Tensor,
) (
	Tensor,
	bool,
) {

	return broadcastBinary(

		t,

		other,

		func(
			a,
			b float32,
		) float32 {

			return a / b

		},
	)

}
func broadcastBinary(
	a Tensor,
	b Tensor,
	op func(
		float32,
		float32,
	) float32,
) (
	Tensor,
	bool,
) {

	outShape, ok :=
		broadcastShape(
			a.Shape(),
			b.Shape(),
		)

	if !ok {

		return Tensor{},
			false

	}

	out :=
		New(
			outShape,
		)

	for i := 0; i < out.NumElements(); i++ {

		av :=
			a.broadcastAt(
				i,
				outShape,
			)

		bv :=
			b.broadcastAt(
				i,
				outShape,
			)

		out.FlatSet(
			i,
			op(
				av,
				bv,
			),
		)

	}

	return out,
		true

}
