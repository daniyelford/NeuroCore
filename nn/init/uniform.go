package init

import (
	"math/rand"

	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Uniform struct {
	Min float32

	Max float32
}

func (u Uniform) Init(
	t *tensor.Tensor,
) {

	for i := 0; i < t.Len(); i++ {

		v :=
			u.Min +
				rand.Float32()*
					(u.Max-u.Min)

		t.FlatSet(
			i,
			v,
		)

	}

}
