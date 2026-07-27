/*
Package autograd implements automatic differentiation.
*/
package autograd

import "github.com/daniyelford/neurocore/internal/core/tensor"

var defaultEngine = NewEngine()
var gradEnabled = true

func CreateNode(op Operation, output *Variable, parents ...*Node) *Node {
	node := output.Node()
	node.Op = op
	node.Parents = parents
	return node
}
func Accumulate(v *Variable, grad tensor.Tensor) {
	current := v.Grad()
	if current.Empty() {
		v.SetGrad(grad)
		return
	}
	v.SetGrad(current.Add(grad))
}
func Backward(v *Variable) {
	root := v.Node()
	nodes := TopologicalSort(root)
	root.Grad = tensor.New(root.Data.Shape())
	root.Grad.Fill(1)
	for i := len(nodes) - 1; i >= 0; i-- {
		node := nodes[i]
		if node.Op == nil {
			continue
		}
		grads, err := node.Op.Backward(node.Grad)
		if err != nil {
			panic(err)
		}
		for index, parent := range node.Parents {
			if !parent.RequiresGrad {
				continue
			}
			Accumulate(VariableFromNode(parent), grads[index])
		}
	}
}
func NewContext() *Context {
	return &Context{}
}
func (c *Context) Add(n *Node) {
	c.nodes = append(c.nodes, n)
}
func (c *Context) Nodes() []*Node {
	return c.nodes
}
func Default() *Engine {
	return defaultEngine
}
func (v *Variable) Detach() *Variable {
	return NewVariable(v.Data(), false)
}
func NewEngine() *Engine {
	return &Engine{graph: NewGraph()}
}
func (e *Engine) Execute(op Operation, inputs ...*Variable) (*Variable, error) {
	out, err := op.Forward(inputs...)
	if err != nil {
		return nil, err
	}
	node := out.Node()
	node.Op = op
	node.Parents = make([]*Node, 0, len(inputs))
	for _, v := range inputs {
		node.Parents = append(node.Parents, v.Node())
	}
	e.graph.Add(node)
	return out, nil
}
func ZeroLike(v Variable) tensor.Tensor {
	return tensor.New(v.Data().Shape())
}
func HasGradient(v Variable) bool {
	return !v.Grad().Empty()
}
func NewGraph() *Graph {
	return &Graph{}
}
func (g *Graph) Add(node *Node) {
	g.nodes = append(g.nodes, node)
}
func (g *Graph) Clear() {
	g.nodes = nil
}
func NewNode(data tensor.Tensor, requiresGrad bool, op Operation, parents ...*Node) *Node {
	return &Node{
		Data:         data,
		RequiresGrad: requiresGrad,
		Op:           op,
		Parents:      parents,
	}
}
func EnableGrad() {
	gradEnabled = true
}
func DisableGrad() {
	gradEnabled = false
}
func GradEnabled() bool {
	return gradEnabled
}
func (v *Variable) ZeroGrad() {
	if v.node.Grad.Empty() {
		return
	}
	v.node.Grad.Zero()
}
func TopologicalSort(root *Node) []*Node {
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
		result = append(result, n)
	}
	visit(root)
	return result
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
func NewVariable(data tensor.Tensor, requiresGrad bool) *Variable {
	return &Variable{node: &Node{Data: data, RequiresGrad: requiresGrad}}
}
func VariableFromNode(n *Node) *Variable {
	return &Variable{node: n}
}
