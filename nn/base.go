package nn

type BaseModule struct {
	training bool
}

func NewBaseModule() BaseModule {

	return BaseModule{

		training: true,
	}

}

func (b *BaseModule) Train() {

	b.training = true

}

func (b *BaseModule) Eval() {

	b.training = false

}

func (b BaseModule) Training() bool {

	return b.training

}

func (b BaseModule) Children() []Module {

	return nil

}
