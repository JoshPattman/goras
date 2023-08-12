// This example uses a convolutional neural network like the previous one, but this time to classify if an image is a dog or a cat.
// We will be using quite a big dataset, which you can either get from:
//
//	Kaggle: https://www.kaggle.com/c/dogs-vs-cats
//	My Google Drive: https://drive.google.com/file/d/1ML0qzboqLncaE_NBz9-ECXffpwWC4Fkx/view?usp=sharing
//
// When unzipped, there should be a single folder ./dataset which has many cat images ./dataset/cat_1.jpg, ... and dog images ./dataset/dog_1.jpg, ...
// This example also serves to show you how you could load images into a tensor for training.
package main

import (
	"fmt"
	"image"
	"os"

	K "github.com/JoshPattman/goras"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

func main() {
	// Find every filename in each class
	dogsNames, catsNames := getAllDogsCatsNames("./dataset")
	fmt.Print("Number of dogs: ", len(dogsNames), "\n")
	fmt.Print("Number of cats: ", len(catsNames), "\n")

	// We are going to use a smaller subset of the data, as there is a lot
	dogsNames = dogsNames[:512]
	catsNames = catsNames[:512]

	// Lets load evey image into an image.Image, and resize them as we go.
	// We are simply stretching the images at the moment. It might be more intelligent to crop them.
	dogImgs := make([]image.Image, 0)
	catImgs := make([]image.Image, 0)

	for _, dogName := range dogsNames {
		img, err := loadImageFromFile(dogName)
		if err != nil {
			panic(err)
		}
		imgResized := K.ImageUtils.ResizeImage(img, 64, 64, "nearest_neighbor")
		dogImgs = append(dogImgs, imgResized)
	}
	fmt.Println("Loaded all dogs images")

	for _, catName := range catsNames {
		img, err := loadImageFromFile(catName)
		if err != nil {
			panic(err)
		}
		imgResized := K.ImageUtils.ResizeImage(img, 64, 64, "nearest_neighbor")
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

	// Now convert all the images to T.Tensor. We have structured this tensor so that it goes dog,cat,dog,cat to make sure that each batch has the same number of dogs and cats
	x, y := K.ImageUtils.ImagesToTensor(datasetImgs), T.New(T.WithShape(len(datasetYs), 1), T.WithBacking(datasetYs))
	fmt.Println("Converted images to tensors")

	model := MakeModel()
	fmt.Println("Built model")

	fmt.Println(model.Summary())

	/*OUTPUT
	Layer 0     model_1::input                Shape: (32, 3, 64, 64)      From: [] Num Params 0
	Layer 1     model_2::conv2d               Shape: (32, 16, 64, 64)     From: [model_1.input       ] Num Params 432
	Layer 2     model_3::activation(relu)     Shape: (32, 16, 64, 64)     From: [model_2.conv        ] Num Params 0
	Layer 3     model_4::maxpool2d            Shape: (32, 16, 32, 32)     From: [model_3.activation  ] Num Params 0
	Layer 4     model_5::conv2d               Shape: (32, 32, 32, 32)     From: [model_4.maxpool     ] Num Params 4608
	Layer 5     model_6::activation(relu)     Shape: (32, 32, 32, 32)     From: [model_5.conv        ] Num Params 0
	Layer 6     model_7::maxpool2d            Shape: (32, 32, 16, 16)     From: [model_6.activation  ] Num Params 0
	Layer 7     model_8::conv2d               Shape: (32, 64, 16, 16)     From: [model_7.maxpool     ] Num Params 18432
	Layer 8     model_9::activation(relu)     Shape: (32, 64, 16, 16)     From: [model_8.conv        ] Num Params 0
	Layer 9    model_10::maxpool2d            Shape: (32, 64, 8, 8)       From: [model_9.activation  ] Num Params 0
	Layer 10   model_11::reshape              Shape: (32, 4096)           From: [model_10.maxpool    ] Num Params 0
	Layer 11   model_12::dense                Shape: (32, 128)            From: [model_11.reshape    ] Num Params 524416
	Layer 12   model_13::activation(relu)     Shape: (32, 128)            From: [model_12.matmul     ] Num Params 0
	Layer 13   model_14::dense                Shape: (32, 64)             From: [model_13.activation ] Num Params 8256
	Layer 14   model_15::activation(relu)     Shape: (32, 64)             From: [model_14.matmul     ] Num Params 0
	Layer 15   model_16::dense                Shape: (32, 1)              From: [model_15.activation ] Num Params 65
	Layer 16   model_17::activation(sigmoid)  Shape: (32, 1)              From: [model_16.matmul     ] Num Params 0
	Total number of parameters: 556209
	*/

	// If this function causes some panic from gorgonia, please raise an issue on the Goras repo. I'm not sure if it is jsut my laptop or if there is a bug.
	solver := G.NewAdamSolver(G.WithLearnRate(0.001))
	model.Fit(K.V(x), K.V(y), solver, K.WithEpochs(14))

	// Ok, now we are going to use some of the images that we trained on (it would be better to use images we have never seen before),
	// And we will get the model to put images it thinks are dogs in the ./dogs dir, and cats in the ./cats dir.
	// When I ran this, the model only got one or two wrong, which is cool!

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
	testX := K.ImageUtils.ImagesToTensor(testSet)
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

	// Finally, lets save the model for another time
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
