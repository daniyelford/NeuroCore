package tensor

type TensorImpl struct {

	storage *storage.Storage

	shape *shape.Shape

	stride *stride.Stride

	offset int

	version uint64
}