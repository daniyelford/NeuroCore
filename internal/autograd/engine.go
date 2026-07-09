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
) Variable {

	out := op.Forward()

	return out

}
