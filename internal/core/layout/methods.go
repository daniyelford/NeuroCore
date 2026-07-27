/*
Package layout defines tensor memory layouts.
*/
package layout

import "strings"

const (
	Unknown Order = iota
	RowMajor
	ColumnMajor
)

func Default() Order {
	return RowMajor
}
func (o Order) Valid() bool {
	switch o {
	case RowMajor,
		ColumnMajor:
		return true
	default:
		return false
	}
}
func (o Order) IsRowMajor() bool {
	return o == RowMajor
}
func (o Order) IsColumnMajor() bool {
	return o == ColumnMajor
}
func Parse(s string) (Order, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "row",
		"row-major",
		"rowmajor",
		"c":
		return RowMajor, true
	case "column",
		"column-major",
		"columnmajor",
		"fortran",
		"f":
		return ColumnMajor, true
	default:
		return Unknown, false
	}
}
func (o Order) String() string {
	switch o {
	case RowMajor:
		return "row-major"
	case ColumnMajor:
		return "column-major"
	default:
		return "unknown"
	}
}
