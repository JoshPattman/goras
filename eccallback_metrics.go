package goras

func CustomMetricCallback(metricFunc func() (float64, error), name string, every int) EpochCallback {
	return func(epoch int, metrics map[string]float64) error {
		if epoch%every == 0 {
			metric, err := metricFunc()
			if err != nil {
				return err
			}
			metrics[name] = metric
		}
		return nil
	}
}
