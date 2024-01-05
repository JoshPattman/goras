package goras

import (
	"fmt"
	"os"
)

// LogCSVEpochMetricsCallback logs the metrics to a file in CSV format.
// It will log as training progresses, and will only close the file when training ends.
func LogCSVEpochMetricsCallback(filename string, metricNames ...string) TrainingCallback {
	var file *os.File

	return TrainingCallback{
		OnTrainingStart: func() error {
			var err error
			file, err = os.Create(filename)
			if err != nil {
				return err
			}
			headerString := "epoch"
			for _, name := range metricNames {
				headerString += "," + name
			}
			_, err = fmt.Fprintf(file, "%v\n", headerString)
			if err != nil {
				return err
			}
			return nil
		},
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
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
				_, err := fmt.Fprintf(file, "%v\n", rowString)
				return false, err
			}
			return false, nil
		},
		OnCleanup: func() {
			file.Close()
		},
	}
}

// LogCSVBatchMetricsCallback logs the metrics to a file in CSV format.
// It will log as training progresses, and will only close the file when training ends.
func LogCSVBatchMetricsCallback(filename string, metricNames ...string) TrainingCallback {
	var file *os.File

	return TrainingCallback{
		OnTrainingStart: func() error {
			var err error
			file, err = os.Create(filename)
			if err != nil {
				return err
			}
			headerString := "epoch,batch,max_batch,adj_epoch"
			for _, name := range metricNames {
				headerString += "," + name
			}
			_, err = fmt.Fprintf(file, "%v\n", headerString)
			if err != nil {
				return err
			}
			return nil
		},
		OnEpochEnd: func(epoch int, metrics map[string]float64) (bool, error) {
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
				_, err := fmt.Fprintf(file, "%v\n", rowString)
				return false, err
			}
			return false, nil
		},
		OnCleanup: func() {
			file.Close()
		},
	}
}