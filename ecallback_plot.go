package goras

import (
	"fmt"
	"io"
)

// LogCSVMetricsCallback logs the metrics to a writer in CSV format.
// It will log as training progresses, and will never close the writer.
func LogCSVMetricsCallback(writer io.Writer, metricNames ...string) EpochCallback {
	headerString := "epoch"
	for _, name := range metricNames {
		headerString += "," + name
	}
	_, err := fmt.Fprintf(writer, "%v\n", headerString)
	if err != nil {
		panic(err)
	}
	return func(epoch int, metrics map[string]float64) error {
		anyMetricsRun := false
		row := make([]string, len(metricNames))
		for i, name := range metricNames {
			if mVal, ok := metrics[name]; ok {
				row[i] = fmt.Sprintf(",%v", mVal)
				anyMetricsRun = true
			}
		}
		if anyMetricsRun {
			rowString := fmt.Sprintf("%v", epoch)
			for _, val := range row {
				rowString += val
			}
			_, err := fmt.Fprintf(writer, "%v\n", rowString)
			return err
		}
		return nil
	}
}
