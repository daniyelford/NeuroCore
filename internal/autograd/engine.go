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
) (*Variable, error) {

	out, err := op.Forward(inputs...)

	if err != nil {
		return nil, err
	}

	return out, nil
}
