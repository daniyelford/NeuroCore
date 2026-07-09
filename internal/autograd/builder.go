package autograd

func CreateNode(
	op Operation,
	output *Variable,
	parents ...*Node,
) *Node {

	node := output.Node()

	node.Op = op
	node.Parents = parents

	return node
}
