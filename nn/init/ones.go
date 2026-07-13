package init

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Ones struct{}

func (o Ones) Init(
	t *tensor.Tensor,
) {

	t.Fill(1)

}
