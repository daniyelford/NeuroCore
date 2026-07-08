package autograd

type Operation interface {
	Forward() Variable

	Backward(
		grad Variable,
	) []Variable

	Name() string
}
