package goras

import T "gorgonia.org/tensor"

type TrainingDataGenerator interface {
	NextBatch(batchSize int) ([]T.Tensor, []T.Tensor, error) // Should return nil, nil if no more data
	Reset(batchSize int) error                               // Resets the generator for the next epoch
	NumBatches() int                                         // Returns the number of batches in this epoch
}

// TensorTrainingDataGenerator is a TrainingDataGenerator that uses tensors as inputs and outputs.
// It shoudl only be used with small datasets, as it requires the entire dataset to be loaded into memory at once.
type TensorTrainingDataGenerator struct {
	inputs                []T.Tensor
	outputs               []T.Tensor
	currentBatchedInputs  [][]T.Tensor
	currentBatchedOutputs [][]T.Tensor
	currentBatch          int
}

// NewTTDG creates a new TensorTrainingDataGenerator.
// This is used by the fit method of the model to generate batches of data.
// The inputs and outputs are the training data and labels respectively.
// They are a slice due to multiple input output capabilities. If you only have one input and output, you can pass in a slice of length 1 for both.
func NewTTDG(xs, ys []T.Tensor) *TensorTrainingDataGenerator {
	return &TensorTrainingDataGenerator{
		inputs:  xs,
		outputs: ys,
	}
}

func (t *TensorTrainingDataGenerator) NextBatch(int) ([]T.Tensor, []T.Tensor, error) {
	if t.currentBatch >= len(t.currentBatchedInputs) {
		return nil, nil, nil
	}
	t.currentBatch++
	return t.currentBatchedInputs[t.currentBatch-1], t.currentBatchedOutputs[t.currentBatch-1], nil
}

func (t *TensorTrainingDataGenerator) Reset(batchSize int) error {
	t.currentBatch = 0
	var err error
	t.currentBatchedInputs, _, err = batchMultipleTensors(t.inputs, batchSize, false)
	if err != nil {
		return err
	}
	t.currentBatchedOutputs, _, err = batchMultipleTensors(t.outputs, batchSize, false)
	if err != nil {
		return err
	}
	return nil
}

func (t *TensorTrainingDataGenerator) NumBatches() int {
	return len(t.currentBatchedInputs)
}
