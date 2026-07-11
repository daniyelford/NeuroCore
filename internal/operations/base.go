package operations

import "github.com/daniyelford/neurocore/internal/autograd"

type Base struct {
	inputs []*autograd.Variable

	output *autograd.Variable
}

func (b *Base) Save(inputs ...*autograd.Variable) {

	b.inputs = append(
		b.inputs[:0],
		inputs...,
	)

}

func (b *Base) Inputs() []*autograd.Variable {

	return b.inputs

}

func (b *Base) Input(i int) *autograd.Variable {

	return b.inputs[i]

}

func (b *Base) Output() *autograd.Variable {

	return b.output

}

func (b *Base) SetOutput(
	v *autograd.Variable,
) {

	b.output = v

}
