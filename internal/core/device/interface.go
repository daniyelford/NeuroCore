package device

type Device interface {

	Type() Type

	Index() int

	Name() string

	IsAvailable() bool

	String() string
}