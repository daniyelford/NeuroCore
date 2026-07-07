package impl

func New(
	st *storage.Storage,
	sh *shape.Shape,
	str *stride.Stride,
	dev device.Device,
) *TensorImpl {

	t := &TensorImpl{
		storage: st,
		shape: sh,
		stride: str,
		device: dev,
	}

	t.version.Store(1)

	return t
}