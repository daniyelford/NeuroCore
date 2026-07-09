package autograd

var defaultEngine = NewEngine()

func Default() *Engine {

	return defaultEngine

}
