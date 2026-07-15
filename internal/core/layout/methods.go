package layout

import "strings"

// Valid reports whether the layout is supported.
func (o Order) Valid() bool {

	switch o {

	case RowMajor,
		ColumnMajor:

		return true

	default:

		return false

	}

}

// IsRowMajor reports whether the layout is C-order.
func (o Order) IsRowMajor() bool {
	return o == RowMajor
}

// IsColumnMajor reports whether the layout is Fortran-order.
func (o Order) IsColumnMajor() bool {
	return o == ColumnMajor
}

// Parse converts a string into an Order.
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
