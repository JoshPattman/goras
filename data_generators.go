package goras

import T "gorgonia.org/tensor"

type TrainingDataGenerator interface {
	NextBatch(batchSize int) (map[string]T.Tensor, map[string]T.Tensor, error) // Should return nil, nil, nil if no more data
	Reset(batchSize int) error                                                 // Resets the generator for the next epoch
	NumBatches() int                                                           // Returns the number of batches in this epoch
}

var _ TrainingDataGenerator = &TensorTrainingDataGenerator{}

// TensorTrainingDataGenerator is a TrainingDataGenerator that uses tensors as inputs and outputs.
// It should only be used with small datasets, as it requires the entire dataset to be loaded into memory at once.
type TensorTrainingDataGenerator struct {
	inputs                map[string]T.Tensor
	outputs               map[string]T.Tensor
	currentBatchedInputs  []map[string]T.Tensor
	currentBatchedOutputs []map[string]T.Tensor
	currentBatch          int
}

// NewTTDG creates a new TensorTrainingDataGenerator.
// This is used by the fit method of the model to generate batches of data.
// The inputs and outputs are the training data and labels respectively.
// They are a slice due to multiple input output capabilities. If you only have one input and output, you can pass in a slice of length 1 for both.
func NewTTDG(xs, ys map[string]T.Tensor) *TensorTrainingDataGenerator {
	return &TensorTrainingDataGenerator{
		inputs:  xs,
		outputs: ys,
	}
}

func (t *TensorTrainingDataGenerator) NextBatch(int) (map[string]T.Tensor, map[string]T.Tensor, error) {
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
