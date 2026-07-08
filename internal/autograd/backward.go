package autograd

func Backward(
	v Variable,
) {

	node := v.Node()

	visited := map[*Node]bool{}

	backwardNode(
		node,
		visited,
	)

}

func backwardNode(
	node *Node,
	visited map[*Node]bool,
) {

	if visited[node] {

		return

	}

	visited[node] = true

	// initial gradient

	node.Output.Grad =
		node.Output.Data

	for _, parent := range node.Parents {

		if parent.Output.RequiresGrad {

			parent.Output.Grad =
				node.Output.Grad

		}

		backwardNode(
			parent,
			visited,
		)

	}

}
