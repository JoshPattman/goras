package main

import (
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
)

// Function to look at a directory and find all files that start with cat or dog, and return the names
func getAllDogsCatsNames(datasetDir string) ([]string, []string) {
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

// Function to load a jpg image from a file
func loadImageFromFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	return img, err
}

// Function to save an image to a jpg file
func saveImageToFile(img image.Image, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, img, nil)
}
