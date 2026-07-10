package autograd

type Engine struct {
	graph *Graph
}

func NewEngine() *Engine {

	return &Engine{

		graph: NewGraph(),
	}

}
func (e *Engine) Execute(
	op Operation,
	inputs ...*Variable,
) (
	*Variable,
	error,
) {

	out, err :=
		op.Forward(
			inputs...,
		)

	if err != nil {

		return nil, err

	}

	node := out.Node()

	node.Op = op

	node.Parents =
		make(
			[]*Node,
			len(inputs),
		)

	for i, v := range inputs {

		node.Parents[i] =
			v.Node()

	}

	e.graph.Add(node)

	return out, nil

}
