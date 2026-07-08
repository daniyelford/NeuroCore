package device

import "sync"

var (
	registry = make(map[string]Device)

	lock sync.RWMutex
)

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
