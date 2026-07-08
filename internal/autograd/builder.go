package autograd

func CreateNode(
	op Operation,
	parents ...*Node,
) *Node {

	node := &Node{

		Op: op,

		Parents: parents,
	}

	node.Output = op.Forward()

	return node

}
