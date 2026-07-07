package version

import "fmt"

func String() string {

	return fmt.Sprintf(
		"%d.%d.%d",
		Major,
		Minor,
		Patch,
	)

}