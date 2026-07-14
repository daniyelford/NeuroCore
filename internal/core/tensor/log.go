package tensor

import "math"

func (t Tensor) Log() Tensor {

	out := New(
		t.Shape(),
	)

	for i := 0; i < t.NumElements(); i++ {

		idx := t.memoryIndex(i)

		out.memory.Set(
			idx,
			float32(
				math.Log(
					float64(
						t.memory.At(idx),
					),
				),
			),
		)

	}

	return out

}
