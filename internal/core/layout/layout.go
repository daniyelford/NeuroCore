package layout

// Order represents a tensor memory layout.
type Order uint8

const (
	Unknown Order = iota

	RowMajor

	ColumnMajor
)
