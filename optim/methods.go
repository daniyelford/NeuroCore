/*
Package optim contains optimization algorithms.
*/
package optim

import (
	"math"

	"github.com/daniyelford/neurocore/internal/core/tensor"
	"github.com/daniyelford/neurocore/nn"
)

func NewSGD(parameters []nn.Parameter, lr float32) *SGD {
	return &SGD{
		Parameters: parameters,
		LR:         lr,
	}
}
func (s *SGD) Step() {
	for _, p := range s.Parameters {
		if p.Value == nil {
			continue
		}
		data := p.Value.Data()
		grad := p.Value.Grad()
		if grad.Empty() {
			panic("parameter gradient empty from optim sgd step")
		}
		if grad.NumElements() != data.NumElements() {
			panic("parameter gradient not initialized")
		}
		for i := 0; i < data.NumElements(); i++ {
			value := data.FlatAt(i)
			g := grad.FlatAt(i)
			data.FlatSet(i, value-s.LR*g)
		}
	}
}
func (s *SGD) ZeroGrad() {
	for _, p := range s.Parameters {
		if p.Value == nil {
			continue
		}
		grad := p.Value.Grad()
		if grad.Empty() {
			continue
		}
		for i := 0; i < grad.NumElements(); i++ {
			grad.FlatSet(i, 0)
		}
	}
}
func NewMomentum(parameters []nn.Parameter, lr float32, beta float32) *Momentum {
	velocity := make([]tensor.Tensor, len(parameters))
	for i, p := range parameters {
		velocity[i] = tensor.New(p.Value.Data().Shape())
	}
	return &Momentum{
		Parameters: parameters,
		LR:         lr,
		Beta:       beta,
		Velocity:   velocity,
	}
}
func (m *Momentum) Step() {
	for i, p := range m.Parameters {
		data := p.Value.Data()
		grad := p.Value.Grad()
		v := m.Velocity[i]
		for j := 0; j < data.NumElements(); j++ {
			old := v.FlatAt(j)
			newV := m.Beta*old + grad.FlatAt(j)
			v.FlatSet(j, newV)
			data.FlatSet(j, data.FlatAt(j)-m.LR*newV)
		}
		m.Velocity[i] = v
	}
}
func (m *Momentum) ZeroGrad() {
	for _, p := range m.Parameters {
		g := p.Value.Grad()
		for i := 0; i < g.NumElements(); i++ {
			g.FlatSet(i, 0)
		}
	}
}
func NewAdam(parameters []nn.Parameter, lr float32) *Adam {
	m := make([]tensor.Tensor, len(parameters))
	v := make([]tensor.Tensor, len(parameters))
	for i, p := range parameters {
		m[i] = tensor.New(p.Value.Data().Shape())
		v[i] = tensor.New(p.Value.Data().Shape())
	}
	return &Adam{
		Parameters: parameters,
		LR:         lr,
		Beta1:      0.9,
		Beta2:      0.999,
		Eps:        1e-8,
		M:          m,
		V:          v,
	}
}
func (a *Adam) Step() {
	a.StepCount++
	for i, p := range a.Parameters {
		data := p.Value.Data()
		grad := p.Value.Grad()
		m := a.M[i]
		v := a.V[i]
		for j := 0; j < data.NumElements(); j++ {
			g := grad.FlatAt(j)
			mValue := a.Beta1*m.FlatAt(j) + (1-a.Beta1)*g
			vValue := a.Beta2*v.FlatAt(j) + (1-a.Beta2)*g*g
			m.FlatSet(j, mValue)
			v.FlatSet(j, vValue)
			mHat := mValue / float32(1-math.Pow(float64(a.Beta1), float64(a.StepCount)))
			vHat := vValue / float32(1-math.Pow(float64(a.Beta2), float64(a.StepCount)))
			update := a.LR * mHat / (float32(math.Sqrt(float64(vHat))) + a.Eps)
			data.FlatSet(j, data.FlatAt(j)-update)
		}
		a.M[i] = m
		a.V[i] = v
	}
}
func (a *Adam) ZeroGrad() {
	for _, p := range a.Parameters {
		g := p.Value.Grad()
		for i := 0; i < g.NumElements(); i++ {
			g.FlatSet(i, 0)
		}
	}
}
