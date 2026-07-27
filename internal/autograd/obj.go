/*
Package autograd implements automatic differentiation.
*/
package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Context struct {
	nodes []*Node
}
type Graph struct {
	nodes []*Node
}
type Engine struct {
	graph *Graph
}
type Variable struct {
	node *Node
}
type Operation interface {
	Name() string
	Forward(inputs ...*Variable) (*Variable, error)
	Backward(grad tensor.Tensor) ([]tensor.Tensor, error)
}
type Node struct {
	Data         tensor.Tensor
	Grad         tensor.Tensor
	RequiresGrad bool
	Parents      []*Node
	Op           Operation
	ID           uint64
}
