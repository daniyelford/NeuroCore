package device

type CUDADevice struct{

	index int

}
func (d CUDADevice) Type() Type{

	return CUDA

}

func (d CUDADevice) Index() int{

	return d.index

}