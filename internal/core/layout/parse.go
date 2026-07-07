package layout

import "strings"

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
