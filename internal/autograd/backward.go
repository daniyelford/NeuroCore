package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

func Backward(
	v Variable,
) {

	root := v.Node()

	nodes :=
		TopologicalSort(root)

	root.Grad =
		tensor.New(
			root.Data.Shape(),
		)

	root.Grad.Fill(1)

	for i := len(nodes) - 1; i >= 0; i-- {

		node := nodes[i]

		if node.Op == nil {

			continue

		}

		grads :=
			node.Op.Backward(
				node.Output.Grad,
			)

		for index, parent := range node.Parents {

			if !parent.Output.RequiresGrad {

				continue

			}

			Accumulate(
				&parent.Output,
				grads[index],
			)

		}

	}

}
