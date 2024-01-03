package goras

// TrainingCallback is a type which specifies some functions that are run during training.
type TrainingCallback struct {
	// OnTrainingStart is called before training starts.
	OnTrainingStart func() error
	// OneEpochEnd is called after each epoch.
	OnEpochEnd func(epoch int, metrics map[string]float64) error
	// OnTrainingEnd is called after training ends.
	OnTrainingEnd func() error
}
