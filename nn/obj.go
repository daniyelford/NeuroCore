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
	Forward(input *autograd.Variable) *autograd.Variable
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
type AvgPool2D struct {
	BaseModule
	KernelH int
	KernelW int
	StrideH int
	StrideW int
}
type AdaptiveAvgPool2D struct {
	BaseModule
	OutputH int
	OutputW int
}
type AdaptiveMaxPool2D struct {
	BaseModule
	OutputH int
	OutputW int
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
	StartDim int
}
type Embedding struct {
	BaseModule
	NumEmbeddings int
	EmbeddingDim  int
	Weight        Parameter
}
type CrossEntropyLoss struct {
	BaseModule

	op operations.CrossEntropyLoss
}
type ConvTranspose2D struct {
	BaseModule
	Weight      Parameter
	Bias        Parameter
	InChannels  int
	OutChannels int
	KernelH     int
	KernelW     int
	StrideH     int
	StrideW     int
	PaddingH    int
	PaddingW    int
}
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
type Dropout struct {
	BaseModule
	P        float32
	Training bool
}
type ReflectionPad2D struct {
	BaseModule
	PadTop    int
	PadBottom int
	PadLeft   int
	PadRight  int
}
type ReplicationPad2D struct {
	BaseModule
	Left   int
	Right  int
	Top    int
	Bottom int
}
type PixelShuffle struct {
	BaseModule
	Scale int
}
type PixelUnshuffle struct {
	BaseModule
	Scale int
}
type RNN struct {
	BaseModule
	InputSize     int
	HiddenSize    int
	NumLayers     int
	BatchFirst    bool
	Bidirectional bool
	Bias          bool
	Nonlinearity  string
	WeightIH      []Parameter
	WeightHH      []Parameter
	BiasIH        []Parameter
	BiasHH        []Parameter
}
