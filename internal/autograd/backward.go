package autograd

func Backward(
	node *Node,
) {

	grad := node.Output.Data

	node.Output.Grad = grad

	for _, parent := range node.Parents {

		if parent.Output.RequiresGrad {

			parent.Output.Grad =
				grad

		}

	}

}
