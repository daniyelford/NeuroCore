package device

type Backend interface{

	Name() string

	Allocate(...)

	Free(...)

	Copy(...)

	Synchronize()

}
type Device interface {

	ID() string

	Name() string

	Backend() Backend

	Memory() uint64

	IsAvailable() bool

}
