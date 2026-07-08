package nn

type Model struct {
	Root Module

	training bool
}

func NewModel(
	module Module,
) *Model {

	return &Model{

		Root: module,

		training: true,
	}

}

func (m *Model) Train() {

	m.training = true

}

func (m *Model) Eval() {

	m.training = false

}
func (m *Model) Parameters() []Parameter {

	return m.Root.Parameters()

}
