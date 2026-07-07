package impl

func (t *TensorImpl) Version() uint64 {

	return t.version.Load()

}

func (t *TensorImpl) IncrementVersion() {

	t.version.Add(1)

}