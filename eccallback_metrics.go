package goras

// CustomEpochMetricCallback creates a callback that runs the given metric function at the end of a training epoch.
// It is only called every `everyNEpochs` epochs.
func CustomEpochMetricCallback(metricFunc func() (float64, error), name string, everyNEpochs int) TrainingCallback {
	return TrainingCallback{
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
			if epoch%everyNEpochs == 0 {
				metric, err := metricFunc()
				if err != nil {
					return false, err
				}
				metrics[name] = metric
			}
			return false, nil
		},
	}
}

// CustomBatchMetricCallback creates a callback that runs the given metric function at the end of a training batch.
// It is only called every `everyNBatches` batches.
func CustomBatchMetricCallback(metricFunc func() (float64, error), name string, everyNBatches int) TrainingCallback {
	return TrainingCallback{
		OnBatchEnd: func(epoch, batch, numBatches int, metrics map[string]float64) error {
			if batch%everyNBatches == 0 {
				metric, err := metricFunc()
				if err != nil {
					return err
				}
				metrics[name] = metric
			}
			return nil
		},
	}
}
