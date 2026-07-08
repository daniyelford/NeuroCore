package device

import "github.com/daniyelford/neurocore/internal/core/backend"

type Type = backend.DeviceType

const (
	Unknown Type = iota

	CPU

	GPU
)
