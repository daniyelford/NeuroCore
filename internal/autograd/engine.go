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

	node.Parents = make([]*Node, 0, len(inputs))

	for _, v := range inputs {

		node.Parents =
			append(
				node.Parents,
				v.Node(),
			)

	}

	e.graph.Add(node)

	return out, nil

}
