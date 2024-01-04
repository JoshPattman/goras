package goras

import (
	"fmt"
	"os"
)

// SaveModelParametersCallback saves the model parameters to the given path.
// It overwrites the file at the given path each epoch, so you only get the most recent model.
func SaveModelParametersCallback(model *Model, path string) TrainingCallback {
	return TrainingCallback{
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
			f, err := os.Create(path)
			if err != nil {
				return false, err
			}
			defer f.Close()
			return false, model.WriteParams(f)
		},
	}
}

// RepeatedSaveModelParametersCallback saves the model parameters to the given path.
// It saves the model every `every` epochs, so you get multiple models.
// The path should contain a %v format specifier, which will be replaced with the epoch number.
func RepeatedSaveModelParametersCallback(model *Model, pathWithFormat string, every int) TrainingCallback {
	return TrainingCallback{
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
			if epoch%every == 0 {
				f, err := os.Create(fmt.Sprintf(pathWithFormat, epoch))
				if err != nil {
					return false, err
				}
				defer f.Close()
				return false, model.WriteParams(f)
			}
			return false, nil
		},
	}
}
