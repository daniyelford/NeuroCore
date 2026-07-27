/*
Package optim contains optimization algorithms.
*/
package optim

import (
	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

type Adam struct {
	Parameters []nn.Parameter
	LR         float32
	Beta1      float32
	Beta2      float32
	Eps        float32
	M          []tensor.Tensor
	V          []tensor.Tensor
	StepCount  int
}
type SGD struct {
	Parameters []nn.Parameter
	LR         float32
}
type Optimizer interface {
	Step()
	ZeroGrad()
}
type Momentum struct {
	Parameters []nn.Parameter
	LR         float32
	Beta       float32
	Velocity   []tensor.Tensor
}
