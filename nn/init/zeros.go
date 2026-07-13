package init

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Zeros struct{}

func (z Zeros) Init(
	t *tensor.Tensor,
) {

	t.Fill(0)

}
