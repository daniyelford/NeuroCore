package nn

import (
	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type Embedding struct {
	BaseModule

	NumEmbeddings int

	EmbeddingDim int

	Weight Parameter
}

func NewEmbedding(
	numEmbeddings int,
	embeddingDim int,
) *Embedding {

	w :=
		tensor.New(
			shape.New(
				numEmbeddings,
				embeddingDim,
			),
		)

	return &Embedding{

		BaseModule: NewBaseModule(),

		NumEmbeddings: numEmbeddings,

		EmbeddingDim: embeddingDim,

		Weight: NewParameter(
			autograd.NewVariable(
				w,
				true,
			),
		),
	}
}

func (e *Embedding) Name() string {

	return "Embedding"

}

func (e *Embedding) Parameters() []Parameter {

	return []Parameter{
		e.Weight,
	}

}

func (e *Embedding) StateDict() map[string]*autograd.Variable {

	return map[string]*autograd.Variable{

		"weight": e.Weight.Value,
	}
}

func (e *Embedding) Forward(
	input autograd.Variable,
) autograd.Variable {

	indices, err :=
		input.Data().Indices()

	if err != nil {

		panic(err)

	}

	out :=
		tensor.New(
			shape.New(
				len(indices),
				e.EmbeddingDim,
			),
		)

	for i, index := range indices {

		if index < 0 ||
			index >= e.NumEmbeddings {

			panic(
				"embedding index out of range",
			)

		}

		for j := 0; j < e.EmbeddingDim; j++ {

			v :=
				e.Weight.Value.Data().At(
					index,
					j,
				)

			out.Set(
				v,
				i,
				j,
			)

		}
	}

	return *autograd.NewVariable(
		out,
		input.RequiresGrad(),
	)

}
