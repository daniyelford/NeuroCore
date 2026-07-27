package examples

// examples/future_api_test.go

// func TestFutureAPI(t *testing.T) {

// 	trainLoader := dataset.NewDataLoader(nil, 32)
// 	trainer :=
// 		training.
// 			New(model).
// 			Optimizer(
// 				optim.Adam(1e-3),
// 			).
// 			Loss(
// 				nn.CrossEntropy(),
// 			).
// 			Epochs(100)

// 	trainer.Run(trainLoader)
// 	model := nn.Sequential(
// 		nn.Linear(2, 64),
// 		nn.ReLU(),
// 		nn.Linear(64, 64),
// 		nn.ReLU(),
// 		nn.Linear(64, 10),
// 	)

// 	trainer := training.NewTrainer(
// 		model,
// 		optim.Adam(1e-3),
// 	)

// 	trainer.Fit(trainLoader, 100)

// 	trainer.Save("model.nc")

// 	loaded, err := training.Load("model.nc")
// 	if err != nil {
// 		t.Fatal(err)
// 	}

// 	prediction := loaded.Predict(nil)

// 	if prediction == nil {
// 		t.Fatal()
// 	}

// }
// func TestAAA(t *testing.T) {
// 	train :=
// 		dataset.
// 			LoadJSON("examples/json_classifier/data.json").
// 			Batch(32).
// 			Shuffle().
// 			Prefetch(2)

// 	model :=
// 		nn.New().
// 			Linear(2, 128).
// 			ReLU().
// 			Dropout(0.2).
// 			Linear(128, 64).
// 			ReLU().
// 			Linear(64, 2)

// 	trainer :=
// 		training.New(model).
// 			Optimizer(
// 				optim.Adam(1e-3),
// 			).
// 			Loss(
// 				nn.CrossEntropy(),
// 			).
// 			Metrics(
// 				training.Accuracy(),
// 			)

// 	history, err := trainer.Fit(train, 100)
// 	if err != nil {
// 		panic(err)
// 	}

//		trainer.Save("classifier.nc")
//	}
// func TestF(t *testing.T) {
// 	trainX, trainY, err :=
// 		dataset.LoadJSON(
// 			"./json_classifier/data.json",
// 		)

// 	if err != nil {
// 		t.Fatal(err)
// 	}

// 	ds :=
// 		dataset.NewTensorDataset(
// 			trainX,
// 			trainY,
// 		)

// 	loader :=
// 		dataset.NewDataLoader(
// 			ds,
// 			32,
// 		)

// 		// model :=
// 		// 	nn.NewSequential(
// 		// 		nn.NewLinear(2, 64),

// 		// 		activation.NewReLU(),
// 		// 		nn.NewLinear(64, 64),
// 		// 		activation.NewReLU(),
// 		// 		nn.NewLinear(64, 2),
// 		// 	)
// 	model := nn.NewSequential(
// 		nn.NewLinear(2, 64),
// 		nn.NewDropout(0.5),
// 		nn.NewLinear(64, 10),
// 	)
// 	trainer :=
// 		training.NewTrainer(
// 			nn.BaseModule(model).Training(),
// 			optim.NewAdam(
// 				model.Parameters(),
// 				1e-3,
// 			),
// 			nn.NewCrossEntropyLoss(),
// 		)

// 	history :=
// 		trainer.Run(
// 			loader,
// 			100,
// 		)

// 	if history.Epochs[100].Train.Loss > 0.1 {
// 		t.Fatal()
// 	}

// 	err =
// 		checkpoint.Save(
// 			&nn.Model{},
// 			"classifier.nc",
// 		)

// 	if err != nil {
// 		t.Fatal(err)
// 	}
// }
