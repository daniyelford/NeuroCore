package autograd

type Context struct {
	nodes []*Node
}

func NewContext() *Context {

	return &Context{}

}

func (c *Context) Add(
	n *Node,
) {

	c.nodes =
		append(
			c.nodes,
			n,
		)

}

func (c *Context) Nodes() []*Node {

	return c.nodes

}
