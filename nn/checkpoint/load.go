package checkpoint

import (
	"encoding/json"
	"os"

	"github.com/daniyelford/neurocore/nn"
)

func Load(
	model *nn.Model,
	path string,
) error {

	data, err :=
		os.ReadFile(path)

	if err != nil {

		return err

	}

	var file File

	err =
		json.Unmarshal(
			data,
			&file,
		)

	if err != nil {

		return err

	}

	state :=
		model.StateDict()

	for name, tensorData := range file.Parameters {

		variable, ok :=
			state[name]

		if !ok {

			continue

		}

		// مقداردهی tensor
		for i, value := range tensorData.Data {

			variable.Data().
				FlatSet(
					i,
					value,
				)

		}

	}

	return nil

}
