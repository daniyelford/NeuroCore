package autograd

type Graph struct {
	nodes []*Node
}

func NewGraph() *Graph {

	return &Graph{}

}

func (g *Graph) Add(
	node *Node,
) {

	g.nodes = append(
		g.nodes,
		node,
	)

}
