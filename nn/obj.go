/*
Package nn contains neural network layers.
*/
package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/internal/operations"
)

type Conv2D struct {
	BaseModule
	Weight      Parameter
	Bias        Parameter
	InChannels  int
	OutChannels int
	KernelH     int
	KernelW     int
	Stride      int
	Padding     int
}
type StateDict map[string]*autograd.Variable
type SerializedTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type SerializedModel struct {
	Parameters map[string]SerializedTensor `json:"parameters"`
}
type Parameter struct {
	Value *autograd.Variable
	Name  string
}
type jsonVariable struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type BaseModule struct {
	training bool
	name     string
}
type Sequential struct {
	BaseModule
	Modules []Module
}
type MSELoss struct {
	BaseModule
}
type Module interface {
	Forward(input autograd.Variable) autograd.Variable
	Parameters() []Parameter
	StateDict() map[string]*autograd.Variable
	Name() string
	Train()
	Eval()
	Children() []Module
}
type Model struct {
	module   Module
	training bool
}
type MaxPool2D struct {
	BaseModule
	KernelH int
	KernelW int
	StrideH int
	StrideW int
}
type Linear struct {
	BaseModule
	Weight Parameter
	Bias   Parameter
	In     int
	Out    int
}
type Flatten struct {
	BaseModule
}
type Embedding struct {
	BaseModule
	NumEmbeddings int
	EmbeddingDim  int
	Weight        Parameter
}
type Dropout struct {
	BaseModule

	Probability float32
}
type CrossEntropyLoss struct {
	BaseModule

	op operations.CrossEntropyLoss
}

//	type Conv2D struct {
//		BaseModule
//		InChannels  int
//		OutChannels int
//		KernelH     int
//		KernelW     int
//		StrideH     int
//		StrideW     int
//		PaddingH    int
//		PaddingW    int
//		Weight      Parameter
//		Bias        Parameter
//	}
type BatchNorm struct {
	BaseModule
	NumFeatures int
	Eps         float32
	Momentum    float32
	Weight      Parameter
	Bias        Parameter
	RunningMean tensor.Tensor
	RunningVar  tensor.Tensor
}
type BatchNorm2D struct {
	BaseModule
	Gamma       Parameter
	Beta        Parameter
	RunningMean tensor.Tensor
	RunningVar  tensor.Tensor
	Channels    int
	Eps         float32
	Momentum    float32
}
type LayerNorm struct {
	BaseModule
	Shape shape.Shape
	Gamma Parameter
	Beta  Parameter
	Eps   float32
}
