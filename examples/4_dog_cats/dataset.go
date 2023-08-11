package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"

	T "gorgonia.org/tensor"
)

func getAllDogsCatsNames(datasetDir string) ([]string, []string) {
	// Find all filenames in the dataset directory
	filenames, err := filepath.Glob(datasetDir + "/*")
	if err != nil {
		panic(err)
	}
	dogs := make([]string, 0)
	cats := make([]string, 0)
	for _, filename := range filenames {
		if filepath.Base(filename)[:3] == "dog" {
			dogs = append(dogs, filename)
		} else if filepath.Base(filename)[:3] == "cat" {
			cats = append(cats, filename)
		}
	}
	return dogs, cats
}

func loadImageFromFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	return img, err
}

func saveImageToFile(img image.Image, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, img, nil)
}

func resizeImage(img image.Image, width, height int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.NearestNeighbor.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	return dst
}

func imagesToTensor(imgs []image.Image) T.Tensor {
	xDim, yDim := imgs[0].Bounds().Size().X, imgs[0].Bounds().Size().Y
	data := []float64{}
	for _, img := range imgs {
		for channel := 0; channel < 3; channel++ {
			for x := 0; x < xDim; x++ {
				for y := 0; y < yDim; y++ {
					r, g, b, _ := img.At(x, y).RGBA()
					r, g, b = r>>8, g>>8, b>>8
					v := 0.0
					switch channel {
					case 0:
						v = float64(r) / 255.0
					case 1:
						v = float64(g) / 255.0
					case 2:
						v = float64(b) / 255.0
					}
					data = append(data, (v))
				}
			}
		}
	}
	return T.New(T.WithShape(len(imgs), 3, xDim, yDim), T.WithBacking(data))
}

func tensorToImages(tens T.Tensor) []image.Image {
	xDim, yDim := tens.Shape()[2], tens.Shape()[3]
	imgs := make([]image.Image, tens.Shape()[0])
	for i := 0; i < tens.Shape()[0]; i++ {
		img := image.NewRGBA(image.Rect(0, 0, xDim, yDim))
		for channel := 0; channel < 3; channel++ {
			for x := 0; x < xDim; x++ {
				for y := 0; y < yDim; y++ {
					v, err := tens.At(i, channel, x, y)
					vi := v.(float64)
					if err != nil {
						panic(err)
					}
					r, g, b, _ := img.At(x, y).RGBA()
					r, g, b = r>>8, g>>8, b>>8
					r8, g8, b8 := uint8(r), uint8(g), uint8(b)
					switch channel {
					case 0:
						img.Set(x, y, color.RGBA{uint8(vi * 255), g8, b8, 255})
					case 1:
						img.Set(x, y, color.RGBA{r8, uint8(vi * 255), b8, 255})
					case 2:
						img.Set(x, y, color.RGBA{r8, g8, uint8(vi * 255), 255})
					}
				}
			}
		}
		imgs[i] = img
	}
	return imgs
}
