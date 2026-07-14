package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/operations"
)

type CrossEntropyLoss struct {
	BaseModule

	op operations.CrossEntropyLoss
}

func NewCrossEntropyLoss() *CrossEntropyLoss {

	return &CrossEntropyLoss{

		BaseModule: NewBaseModule(),
	}

}
func (c *CrossEntropyLoss) Name() string {

	return "CrossEntropyLoss"

}
func (c *CrossEntropyLoss) Parameters() []Parameter {

	return []Parameter{}

}
func (c *CrossEntropyLoss) Forward(
	input autograd.Variable,
	target autograd.Variable,
) autograd.Variable {

	out, err :=
		c.op.Forward(
			&input,
			&target,
		)

	if err != nil {

		panic(err)

	}

	return *out

}
