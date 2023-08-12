package goras

import (
	"image"
	"image/color"

	"golang.org/x/image/draw"
	T "gorgonia.org/tensor"
)

// ImageUtils is a struct that contains functions that are not core to goras, but are useful for image manipulation.
var ImageUtils imageUtils = imageUtils{}

type imageUtils struct{}

// ImagesToTensor converts a list of image.Image to a tensor.
// Every image should have the same number of channels and the same dimensions.
// TODO : currently this only works for 3 channels. Make it work for b&w images.
func (imageUtils) ImagesToTensor(imgs []image.Image) T.Tensor {
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

// TensorToImages converts a tensor to a list of image.Image.
// The tensor should have the shape (n, 3, x, y).
// TODO : currently this only works for 3 channels. Make it work for b&w images.
func (imageUtils) TensorToImages(tens T.Tensor) []image.Image {
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

// ResizeImage streches or sqeezes an image to a certain size. It uses the specified interpolation, which is one of "nearest_neighbor", "bilinear" or "approx_bilinear
func (imageUtils) ResizeImage(img image.Image, width, height int, interpolation string) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	var interpolator draw.Interpolator
	switch interpolation {
	case "nearest_neighbor":
		interpolator = draw.NearestNeighbor
	case "bilinear":
		interpolator = draw.BiLinear
	case "approx_bilinear":
		interpolator = draw.ApproxBiLinear
	default:
		panic("unrecognized interpolation method")
	}
	interpolator.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	return dst
}

/*
// TransformImage rotates and/or scales an image. It uses the specified interpolation, which is one of "nearest_neighbor", "bilinear" or "approx_bilinear
func (imageUtils) TransformImage(img image.Image, rotationDegrees, scale float64, interpolation string) image.Image {
	panic("not implemented yet")
}*/
