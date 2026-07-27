package operations

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Base struct {
	inputs []*autograd.Variable
	output *autograd.Variable
}
type Transpose struct {
	Base
}
type Tanh struct {
	Base
}
type Sum struct {
	Base
}
type Add struct {
	Base
}
type Sub struct {
	Base
}
type Sigmoid struct {
	Base
}
type Reshape struct {
	Base
	newShape shape.Shape
}
type ReLU struct {
	Base
}
type Neg struct {
	Base
}
type Mul struct {
	Base
}
type MatMul struct {
	Base
}
type Flatten struct {
	Base
}
type Div struct {
	Base
}
type MSE struct {
	Base
}
type Mean struct {
	Base
}
type CrossEntropyLoss struct {
	Base
}
type BatchNorm struct {
	Base
	Eps      float32
	Channels int
	Mean     tensor.Tensor
	Variance tensor.Tensor
}
type MaxPool2D struct {
	Base
	KernelH int
	KernelW int
	StrideH int
	StrideW int
	ArgMax  []int
}
type Conv2D struct {
	Base
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
	KernelH  int
	KernelW  int
}
type LeakyReLU struct {
	NegativeSlope float32
}
