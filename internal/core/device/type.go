package device

// Type represents a compute device type.
type Type uint8

const (
	Unknown Type = iota

	CPU

	GPU
)
