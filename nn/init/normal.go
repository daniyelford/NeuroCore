package init

import (
	"math"
	"math/rand"

	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Normal struct {
	Mean float32

	Std float32
}

func (n Normal) Init(
	t *tensor.Tensor,
) {

	for i := 0; i < t.Len(); i++ {

		u1 := rand.Float32()
		u2 := rand.Float32()

		z :=
			float32(
				math.Sqrt(
					-2*math.Log(
						float64(u1),
					),
				) *
					math.Cos(
						2*math.Pi*float64(u2),
					),
			)

		t.FlatSet(
			i,
			n.Mean+n.Std*z,
		)
	}

}
