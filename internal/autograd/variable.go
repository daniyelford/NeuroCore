package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

type Variable struct {
	node *Node
}

func (v *Variable) Node() *Node {
	return v.node
}
func (v *Variable) Data() tensor.Tensor {
	return v.node.Data
}
func (v *Variable) Grad() tensor.Tensor {
	return v.node.Grad
}
func (v *Variable) SetGrad(g tensor.Tensor) {
	v.node.Grad = g
}
func (v *Variable) RequiresGrad() bool {
	return v.node.RequiresGrad
}
func (v *Variable) SetRequiresGrad(b bool) {
	v.node.RequiresGrad = b
}
func (v *Variable) SetData(t tensor.Tensor) {
	v.node.Data = t
}
func NewVariable(
	data tensor.Tensor,
	requiresGrad bool,
) *Variable {

	return &Variable{

		node: &Node{

			Data: data,

			RequiresGrad: requiresGrad,
		},
	}
}
func VariableFromNode(
	n *Node,
) *Variable {

	return &Variable{
		node: n,
	}

}
