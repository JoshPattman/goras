package goras

// EpochCallback is a function that is called after each epoch.
// It is passed the epoch number and the metrics for that epoch.
// If it returns an error, training will stop.
// metrics will always at least contain the "loss" metric, however other metrics will added only if they were run this epoch.
type EpochCallback func(epoch int, metrics map[string]float64) error
