package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
)

type MSELoss struct {
	BaseModule
}

func NewMSELoss() *MSELoss {

	return &MSELoss{

		BaseModule: NewBaseModule(),
	}

}

func (m *MSELoss) Name() string {

	return "MSELoss"

}

func (m *MSELoss) Parameters() []Parameter {

	return nil

}

func (m *MSELoss) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}

func (m *MSELoss) Forward(
	prediction autograd.Variable,
	target autograd.Variable,
) autograd.Variable {

	op := &operations.MSELoss{}

	out, err := op.Forward(

		&prediction,

		&target,
	)

	if err != nil {

		panic(err)

	}

	return *out

}

//	type Loss interface {
//		Forward(
//			prediction *autograd.Variable,
//			target *autograd.Variable,
//		) *autograd.Variable
//	}

// type MSE struct{}

// func NewMSE() MSE {

// 	return MSE{}

// }

// func (m MSE) Forward(
// 	prediction *autograd.Variable,
// 	target *autograd.Variable,
// ) *autograd.Variable {

// 	diff :=
// 		prediction.Data().Sub(
// 			target.Data(),
// 		)

// 	sum := float32(0)

// 	for i := 0; i < diff.NumElements(); i++ {

// 		v := diff.At(i)

// 		sum += v * v

// 	}

// 	result :=
// 		sum /
// 			float32(diff.NumElements())

// 	out := tensor.New(
// 		shape.New(1),
// 	)

// 	out.Set(
// 		result,
// 		0,
// 	)

// 	return autograd.NewVariable(
// 		out,
// 		true,
// 	)

// }
