package memory

// View represents a window into existing memory.
type View struct {
	memory *Memory

	offset int

	length int
}

func (v View) Len() int {

	return v.length

}

func (v View) At(index int) float32 {

	return v.memory.data[v.offset+index]

}

func (v View) Set(index int, value float32) {

	v.memory.data[v.offset+index] = value

}
