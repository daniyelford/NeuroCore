package init

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Ones struct{}
type Xavier struct{}
type Zeros struct{}
type Kaiming struct{}
type Initializer interface {
	Init(t *tensor.Tensor)
}
type Normal struct {
	Mean float32
	Std  float32
}
type Uniform struct {
	Min float32
	Max float32
}
