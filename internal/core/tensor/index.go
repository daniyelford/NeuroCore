package tensor

import "fmt"

func (t Tensor) Indices() ([]int, error) {

	out := make([]int, t.Len())

	for i := 0; i < t.Len(); i++ {

		v := t.FlatAt(i)

		index := int(v)

		if float32(index) != v {

			return nil, fmt.Errorf(
				"tensor value %v is not an integer index",
				v,
			)
		}

		out[i] = index
	}

	return out, nil
}
