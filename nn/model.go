package nn

import (
	"fmt"

	"github.com/daniyelford/neurocore/internal/autograd"
)

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
func (m *Model) Name() string {

	return "Model"

}
func (m *Model) Train() {
	m.training = true
	m.Root.Train()
}

func (m *Model) Eval() {
	m.training = false
	m.Root.Eval()
}
func (m *Model) Parameters() []Parameter {

	return m.Root.Parameters()

}
func (m *Model) Forward(
	input autograd.Variable,
) autograd.Variable {

	return m.Root.Forward(
		input,
	)

}
func (m *Model) StateDict() map[string]*autograd.Variable {

	return m.Root.StateDict()

}
func (m *Model) LoadStateDict(
	state map[string]*autograd.Variable,
) error {

	current := m.StateDict()

	for name, value := range state {

		dst, ok := current[name]
		if !ok {

			return fmt.Errorf(
				"unknown parameter: %s",
				name,
			)

		}

		dst.SetData(
			value.Data().Clone(),
		)

		dst.SetRequiresGrad(
			value.RequiresGrad(),
		)

	}

	return nil
}

// func (m *Model) Save(
// 	path string,
// ) error {
// 	state := m.StateDict()

// 	model := SerializedModel{

// 		Parameters: map[string]SerializedTensor{},
// 	}

// 	for name, value := range state {

// 		model.Parameters[name] =
// 			serializeTensor(
// 				value.Data(),
// 			)

// 	}
// 	file, err := os.Create(path)
// 	if err != nil {
// 		return err
// 	}
// 	defer file.Close()

// 	encoder := json.NewEncoder(file)

// 	return encoder.Encode(model)
// }
// func (m *Model) Load(
// 	path string,
// ) error {
// 	file, _ := os.Open(path)
// 	var model SerializedModel

// 	decoder := json.NewDecoder(file)

// 	decoder.Decode(&model)
// 	state := StateDict{}

// 	for name, tensorData := range model.Parameters {

// 		state[name] =
// 			autograd.NewVariable(
// 				deserializeTensor(
// 					tensorData,
// 				),
// 				true,
// 			)

// 	}
// 	return m.LoadStateDict(state)
// }
