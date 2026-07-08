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

	if t.Len() == 1 {

		return t.memory.At(0)

	}

	return t.memory.At(index)

}
