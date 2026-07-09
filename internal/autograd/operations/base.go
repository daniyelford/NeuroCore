package operations

import "github.com/daniyelford/neurocore/internal/autograd"

type Base struct {
	Inputs []autograd.Variable

	Output autograd.Variable
}

func (b *Base) Save(inputs ...autograd.Variable) {

	b.Inputs = inputs

}

func (b *Base) Input(i int) autograd.Variable {

	return b.Inputs[i]

}

func (b *Base) SetOutput(v autograd.Variable) {

	b.Output = v

}

func (b *Base) Result() autograd.Variable {

	return b.Output

}

func (b *Base) ZeroGrad() {

	for i := range b.Inputs {
		b.Inputs[i].ZeroGrad()
	}

}
