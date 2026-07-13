package nn

import (
	"encoding/json"
	"os"

	"github.com/daniyelford/neurocore/internal/autograd"
	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

func LoadJSON(
	path string,
) (StateDict, error) {

	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	raw := map[string]jsonVariable{}

	err = json.Unmarshal(bytes, &raw)
	if err != nil {
		return nil, err
	}

	state := StateDict{}

	for name, j := range raw {

		t := tensor.New(
			shape.New(j.Shape...),
		)

		for i, v := range j.Data {
			t.FlatSet(i, v)
		}

		state[name] =
			autograd.NewVariable(
				t,
				true,
			)
	}

	return state, nil
}
