package tensor

func (t Tensor) Clone() Tensor {

	return Tensor{

		shape: t.shape,

		stride: t.stride,

		memory: t.memory.Clone(),

		device: t.device,

		layout: t.layout,
	}

}
