package tensor

func (t Tensor) ArgMax() int {

	maxIndex := 0

	maxValue :=
		t.FlatAt(0)

	for i := 1; i < t.NumElements(); i++ {

		v :=
			t.FlatAt(i)

		if v > maxValue {

			maxValue = v

			maxIndex = i

		}

	}

	return maxIndex

}
