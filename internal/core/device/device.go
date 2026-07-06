package device

type Type uint8

const (
	Unknown Type = iota

	CPU

	CUDA

	Metal

	ROCm
)