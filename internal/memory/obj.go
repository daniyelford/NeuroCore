/*
Package memory contains the memory subsystem.

# Storage

↓

# Buffer

↓

Backend
*/
package memory

import "github.com/daniyelford/neurocore/internal/core/backend"

type View struct {
	memory *Memory
	offset int
	length int
}
type Memory struct {
	data     []float32
	capacity int
	device   backend.DeviceType
}
