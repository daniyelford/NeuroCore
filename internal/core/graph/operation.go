package graph

type Operation interface {
	Forward() NodeValue

	Backward(
		NodeValue,
	) []NodeValue

	Name() string
}
