package impl

func (t *TensorImpl) Shape() *shape.Shape {
	return t.shape
}

func (t *TensorImpl) Stride() *stride.Stride {
	return t.stride
}

func (t *TensorImpl) Storage() *storage.Storage {
	return t.storage
}

func (t *TensorImpl) Device() device.Device {
	return t.device
}

func (t *TensorImpl) Offset() int {
	return t.offset
}