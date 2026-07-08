package autograd

type Node struct {
	Output Variable

	Parents []*Node

	Op Operation
}
