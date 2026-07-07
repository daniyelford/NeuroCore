package layout

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
