/*
Package device defines compute devices.
*/
package device

import (
	"sync"

	"github.com/daniyelford/neurocore/internal/core/backend"
	"github.com/daniyelford/neurocore/internal/memory"
)

var (
	registry = make(map[string]Device)
	lock     sync.RWMutex
)

const (
	Unknown Type = iota
	CPU
	GPU
)

func NewCPU() CPUDevice {
	return CPUDevice{}
}
func (CPUDevice) Type() Type {
	return CPU
}
func (CPUDevice) Name() string {
	return "cpu"
}
func Default() Device {
	cpu := NewCPU()
	return cpu
}
func init() {
	Register("cpu", NewCPU())
}
func Register(name string, d Device) {
	lock.Lock()
	defer lock.Unlock()
	registry[name] = d
}
func Get(name string) (Device, bool) {
	lock.RLock()
	defer lock.RUnlock()
	d, ok := registry[name]
	return d, ok
}
func (CPUDevice) Allocate(size int) memory.Memory {
	return memory.New(size)
}
func (CPUDevice) Copy(dst memory.Memory, src memory.Memory) {
	memory.Copy(dst, src)
}
func NewContext(d Device) Context {
	return Context{device: d}
}
func DefaultContext() Context {
	return NewContext(Default())
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
func (c Context) Allocate(size int) memory.Memory {
	return c.device.Allocate(size)
}
