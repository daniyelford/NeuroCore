package checkpoint

import (
	"encoding/json"
	"os"

	"github.com/daniyelford/neurocore/nn"
)

func Load(model *nn.Model, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var file File
	err = json.Unmarshal(data, &file)
	if err != nil {
		return err
	}
	state := model.StateDict()
	for name, tensorData := range file.Parameters {
		variable, ok := state[name]
		if !ok {
			continue
		}
		for i, value := range tensorData.Data {
			variable.Data().FlatSet(i, value)
		}
	}
	return nil
}
func Save(model *nn.Model, path string) error {
	file := File{Parameters: make(map[string]TensorData)}
	for name, variable := range model.StateDict() {
		t := variable.Data()
		values := make([]float32, t.Len())
		for i := 0; i < t.Len(); i++ {
			values[i] = t.FlatAt(i)
		}
		file.Parameters[name] = TensorData{Shape: t.Shape().Values(), Data: values}
	}
	bytes, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}
