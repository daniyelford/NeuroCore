package autograd

type Node struct {
	Output Variable

	Parents []*Node

	Op Operation
}

func (n *Node) RequiresGrad() bool {

	return n.Output.RequiresGrad

}
