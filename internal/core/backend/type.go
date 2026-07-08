package backend

type DeviceType uint8

const (
	Unknown DeviceType = iota

	CPU

	GPU
)
