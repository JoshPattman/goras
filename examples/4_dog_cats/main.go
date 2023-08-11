package main

// YOU NEED TO UNZIP THE dataset.zip FILE FIRST.
// GET THIS FILE FROM https://drive.google.com/file/d/1ML0qzboqLncaE_NBz9-ECXffpwWC4Fkx/view?usp=sharing

import (
	"fmt"
	"image"
	"os"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	dogsNames, catsNames := getAllDogsCatsNames("./dataset")
	fmt.Print("Number of dogs: ", len(dogsNames), "\n")
	fmt.Print("Number of cats: ", len(catsNames), "\n")

	dogsNames = dogsNames[:128]
	catsNames = catsNames[:128]

	dogImgs := make([]image.Image, 0)
	catImgs := make([]image.Image, 0)

	for _, dogName := range dogsNames {
		img, err := loadImageFromFile(dogName)
		if err != nil {
			panic(err)
		}
		imgResized := resizeImage(img, 64, 64)
		dogImgs = append(dogImgs, imgResized)
	}
	fmt.Println("Loaded all dogs images")

	for _, catName := range catsNames {
		img, err := loadImageFromFile(catName)
		if err != nil {
			panic(err)
		}
		imgResized := resizeImage(img, 64, 64)
		catImgs = append(catImgs, imgResized)
	}
	fmt.Println("Loaded all cats images")

	datasetImgs := make([]image.Image, 0)
	datasetYs := make([]float64, 0)
	for i := range catImgs {
		datasetImgs = append(datasetImgs, dogImgs[i])
		datasetYs = append(datasetYs, 0)

		datasetImgs = append(datasetImgs, catImgs[i])
		datasetYs = append(datasetYs, 1)
	}

	x, y := imagesToTensor(datasetImgs), T.New(T.WithShape(len(datasetYs), 1), T.WithBacking(datasetYs))
	fmt.Println("Converted images to tensors")

	model := MakeModel()
	fmt.Println("Built model")

	fmt.Println(model.Summary())

	solver := G.NewAdamSolver(G.WithLearnRate(0.001))
	model.Fit(K.V(x), K.V(y), solver, K.WithEpochs(14), K.WithClearLine(false))

	// Clear the directories
	if err := os.RemoveAll("./dogs"); err != nil {
		panic(err)
	}
	if err := os.RemoveAll("./cats"); err != nil {
		panic(err)
	}
	// Make empty dir call dogs and cats
	if err := os.Mkdir("./dogs", 0777); err != nil {
		panic(err)
	}
	if err := os.Mkdir("./cats", 0777); err != nil {
		panic(err)
	}

	testSet := make([]image.Image, 0)
	testDogs := dogImgs[len(dogImgs)-16:]
	testCats := catImgs[len(dogImgs)-16:]
	for i := range testDogs {
		testSet = append(testSet, testCats[i])
		testSet = append(testSet, testDogs[i])
	}
	testX := imagesToTensor(testSet)
	pred, err := model.PredictBatch(K.V(testX))
	if err != nil {
		panic(err)
	}
	for i := range testSet {
		predVal, err := pred.At(i, 0)
		if err != nil {
			panic(err)
		}
		predicted := predVal.(float64)
		if predicted > 0.5 {
			saveImageToFile(testSet[i], fmt.Sprintf("./cats/%d.png", i))
		} else {
			saveImageToFile(testSet[i], fmt.Sprintf("./dogs/%d.png", i))
		}
	}

	func() {
		file, err := os.Create("model.gob")
		if err != nil {
			panic(err)
		}
		defer file.Close()
		if err := model.WriteParams(file); err != nil {
			panic(err)
		}
	}()
}
