package memory

func New(size int) Memory {

	if size < 0 {
		size = 0
	}

	return Memory{

		data: make([]float32, size),

		capacity: size,
	}

}

func From(values []float32) Memory {

	data := make([]float32, len(values))

	copy(data, values)

	return Memory{

		data: data,

		capacity: len(data),
	}

}
func NewView(
	m *Memory,
	offset int,
	length int,
) View {

	return View{

		memory: m,

		offset: offset,

		length: length,
	}

}
