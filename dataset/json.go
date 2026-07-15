package dataset

import (
	"encoding/json"
	"os"

	"github.com/daniyelford/neurocore/internal/core/shape"
	"github.com/daniyelford/neurocore/internal/core/tensor"
)

type JSONSample struct {
	X []float32 `json:"x"`

	Y int `json:"y"`
}

type JSONFile struct {
	Samples []JSONSample `json:"samples"`
}

func LoadJSON(
	path string,
) (
	tensor.Tensor,
	tensor.Tensor,
	error,
) {

	bytes, err :=
		os.ReadFile(path)

	if err != nil {

		return tensor.Tensor{},
			tensor.Tensor{},
			err

	}

	var file JSONFile

	err =
		json.Unmarshal(
			bytes,
			&file,
		)

	if err != nil {

		return tensor.Tensor{},
			tensor.Tensor{},
			err

	}

	rows :=
		len(file.Samples)

	cols :=
		len(file.Samples[0].X)

	x :=
		tensor.New(
			shape.New(
				rows,
				cols,
			),
		)

	y :=
		tensor.New(
			shape.New(
				rows,
			),
		)

	for i, s := range file.Samples {

		for j, v := range s.X {

			x.Set(
				v,
				i,
				j,
			)

		}

		y.Set(
			float32(s.Y),
			i,
		)

	}

	return x, y, nil

}
