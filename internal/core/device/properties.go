package device

import (
	"sync"

	"github.com/daniyelford/neurocore/internal/core/backend"
)

type Type = backend.DeviceType

var (
	registry = make(map[string]Device)

	lock sync.RWMutex
)

const (
	Unknown Type = iota

	CPU

	GPU
)

func Default() Device {

	cpu := NewCPU()

	return cpu

}
func init() {
	Register(
		"cpu",
		NewCPU(),
	)

}
func Register(
	name string,
	d Device,
) {
	lock.Lock()
	defer lock.Unlock()
	registry[name] = d
}

func Get(
	name string,
) (Device, bool) {
	lock.RLock()
	defer lock.RUnlock()
	d, ok := registry[name]
	return d, ok
}
