package autograd

func TopologicalSort(
	root *Node,
) []*Node {

	result := []*Node{}

	visited := map[*Node]bool{}

	var visit func(*Node)

	visit = func(n *Node) {

		if visited[n] {

			return

		}

		visited[n] = true

		for _, p := range n.Parents {

			visit(p)

		}

		result =
			append(
				result,
				n,
			)

	}

	visit(root)

	return result

}
