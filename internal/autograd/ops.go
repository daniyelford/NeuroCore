package autograd

func Add(
	a Variable,
	b Variable,
) Variable {

	out :=
		a.Data.Add(
			b.Data,
		)

	v :=
		NewVariable(
			out,
			a.RequiresGrad ||
				b.RequiresGrad,
		)

	node := &Node{

		Output: v,

		Parents: []*Node{

			a.Node(),

			b.Node(),
		},
	}

	v.node = node

	return v

}
func Mul(
	a Variable,
	b Variable,
) Variable {

	out :=
		a.Data.Mul(
			b.Data,
		)

	v :=
		NewVariable(
			out,
			a.RequiresGrad ||
				b.RequiresGrad,
		)

	node := &Node{

		Output: v,

		Parents: []*Node{

			a.Node(),

			b.Node(),
		},
	}

	v.node = node

	return v

}
