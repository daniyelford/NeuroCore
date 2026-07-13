package nn

import (
	"encoding/json"
	"os"
)

type jsonVariable struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

func SaveJSON(
	state StateDict,
	path string,
) error {

	out := map[string]jsonVariable{}

	for name, v := range state {

		data := make([]float32, v.Data().Len())

		for i := 0; i < v.Data().Len(); i++ {
			data[i] = v.Data().FlatAt(i)
		}

		out[name] = jsonVariable{
			Shape: v.Data().Shape().Values(),
			Data:  data,
		}
	}

	bytes, err := json.MarshalIndent(
		out,
		"",
		"  ",
	)
	if err != nil {
		return err
	}

	return os.WriteFile(
		path,
		bytes,
		0644,
	)
}
