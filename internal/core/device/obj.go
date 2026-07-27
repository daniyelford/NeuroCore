/*
Package device defines compute devices.
*/
package device

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/memory"
)

type CPUDevice struct{}
type Type = backend.DeviceType
type Device interface {
	Type() Type
	Name() string
	Allocate(size int) memory.Memory
	Copy(dst memory.Memory, src memory.Memory)
}
type Context struct {
	device Device
}
