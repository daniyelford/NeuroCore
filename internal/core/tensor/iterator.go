package tensor

type Iterator struct {
	tensor Tensor

	index int
}

func (t Tensor) Iterator() Iterator {

	return Iterator{

		tensor: t,

		index: 0,
	}

}

func (it *Iterator) HasNext() bool {

	return it.index < it.tensor.Len()

}

func (it *Iterator) Next() float32 {

	v := it.tensor.memory.At(
		it.index,
	)

	it.index++

	return v

}
