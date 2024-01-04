package goras

// TrainingCallback is a type which specifies some functions that are run during training.
type TrainingCallback struct {
	// OnTrainingStart is called before training starts.
	// If it returns an error, training will not start.
	OnTrainingStart func() error
	// OneEpochEnd is called after each epoch.
	// If it returns an error, training will stop.
	// If it returns true, training will stop gracefully with no error.
	OnEpochEnd func(epoch int, metrics map[string]float64) (bool, error)
	// OnTrainingEnd is called after training ends.
	// If it returns an error, other OnTrainingEnd callbacks may not be called, as the fit function will exit.
	OnTrainingEnd func() error
	// OnCleanup is called after training ends, even if there is an error.
	// It should just deal with errors itself, as they will be ignored.
	// OnCleanup will always be called, no matter how the fit function exited.
	OnCleanup func()
}
