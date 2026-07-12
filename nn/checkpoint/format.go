package checkpoint

type TensorData struct {
	Shape []int `json:"shape"`

	Data []float32 `json:"data"`
}

type File struct {
	Parameters map[string]TensorData `json:"parameters"`
}
