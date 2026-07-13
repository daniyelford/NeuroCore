package nn

import (
	"math/rand"

	"github.com/daniyelford/neurocore/internal/autograd"
)

type Dropout struct {
	BaseModule

	Probability float32
}

func NewDropout(
	p float32,
) *Dropout {

	if p < 0 {
		p = 0
	}

	if p > 1 {
		p = 1
	}

	return &Dropout{

		BaseModule: NewBaseModule(),

		Probability: p,
	}
}

func (d *Dropout) Name() string {

	return "Dropout"

}

func (d *Dropout) Parameters() []Parameter {

	return nil

}

func (d *Dropout) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{}

}

func (d *Dropout) Forward(
	input autograd.Variable,
) autograd.Variable {

	// Eval mode
	if !d.Training() {

		return input

	}

	data := input.Data()

	out := data.Clone()

	scale := float32(1.0)

	if d.Probability < 1 {

		scale = 1 / (1 - d.Probability)

	}

	for i := 0; i < out.Len(); i++ {

		if rand.Float32() < d.Probability {

			out.FlatSet(
				i,
				0,
			)

		} else {

			out.FlatSet(
				i,
				out.FlatAt(i)*scale,
			)

		}

	}

	return *autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}
