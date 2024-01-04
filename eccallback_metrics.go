package goras

func CustomMetricCallback(metricFunc func() (float64, error), name string, every int) TrainingCallback {
	return TrainingCallback{
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
			if epoch%every == 0 {
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
