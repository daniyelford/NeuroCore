package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
)

type MaxPool2D struct {
	BaseModule

	KernelH int
	KernelW int

	StrideH int
	StrideW int
}

func NewMaxPool2D(
	kernelH,
	kernelW,
	strideH,
	strideW int,
) *MaxPool2D {

	return &MaxPool2D{

		BaseModule: NewBaseModule(),

		KernelH: kernelH,
		KernelW: kernelW,

		StrideH: strideH,
		StrideW: strideW,
	}

}
func (m *MaxPool2D) Name() string {

	return "MaxPool2D"

}
func (m *MaxPool2D) Forward(
	input autograd.Variable,
) autograd.Variable {

	op :=
		operations.NewMaxPool2D(

			m.KernelH,
			m.KernelW,

			m.StrideH,
			m.StrideW,
		)

	out, err :=
		op.Forward(
			&input,
		)

	if err != nil {

		panic(err)

	}

	return *out

}
func (m *MaxPool2D) Parameters() []Parameter {

	return []Parameter{}

}
func (m *MaxPool2D) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}
