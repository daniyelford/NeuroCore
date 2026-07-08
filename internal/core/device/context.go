package device

import (
	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/memory"
)

type Context struct {
	device Device
}

func NewContext(
	d Device,
) Context {

	return Context{
		device: d,
	}

}

func DefaultContext() Context {

	return NewContext(
		Default(),
	)

}

func (c Context) Device() Device {

	return c.device

}

func (c Context) Type() backend.DeviceType {

	return c.device.Type()

}

func (c Context) Name() string {

	return c.device.Name()

}
func (c Context) Allocate(
	size int,
) memory.Memory {

	return c.device.Allocate(size)

}
