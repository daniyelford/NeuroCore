package layout

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
