package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

func Backward(
	v *Variable,
) {

	root := v.Node()

	nodes :=
		TopologicalSort(root)
	println("nodes:", len(nodes))

	for _, n := range nodes {
		println(
			"ADD NODE OP:",
			n.Op != nil,
		)

		println(
			"ADD PARENTS:",
			len(n.Parents),
		)
		if n.Op != nil {
			println("node op:", n.Op.Name())
		} else {
			println("node leaf")
		}

	}
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
		println(
			"parents:",
			len(node.Parents),
		)

		grads, err :=
			node.Op.Backward(
				node.Grad,
			)
		if err != nil {
			panic(err)
		}
		for index, parent := range node.Parents {
			if !parent.RequiresGrad {
				continue
			}
			// println(
			// 	"before accumulate parent grad len:",
			// 	VariableFromNode(parent).Grad().Len(),
			// )
			Accumulate(
				VariableFromNode(parent),
				grads[index],
			)

			// println(
			// 	"after accumulate parent grad len:",
			// 	VariableFromNode(parent).Grad().Len(),
			// )

		}

	}

}
