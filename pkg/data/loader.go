package data

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// Sample represents a single data point.
type Sample struct {
	X []float64
	Y float64
}

// StreamCSV streams CSV rows as Samples through a channel. The labelCol is the index of label.
// Close the returned done chan to stop early.
func StreamCSV(path string, labelCol int, out chan<- Sample) (done chan struct{}, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(bufio.NewReader(file))
	reader.ReuseRecord = true
	done = make(chan struct{})

	go func() {
		// Close the file when the goroutine finishes, either by EOF or early termination.
		defer file.Close()
		// Close the output channel to signal that no more samples will be sent.
		defer close(out)
		for {
			select {
			case <-done:
				// If a signal is received on the done channel, terminate early.
				return
			default:
				rec, err := reader.Read()
				if err == io.EOF {
					return
				} // End of file, exit the goroutine.
				if err != nil {
					fmt.Printf("Skipping Record DUe to read Error")
					continue
				} // Skip malformed records.

				// Check if labelCol is valid index within the record
				if labelCol < 0 || labelCol >= len(rec) {
					fmt.Printf("Skipping Record: Column out of Bound")
					continue
				}

				x := make([]float64, 0, len(rec)-1)
				var y float64
				validRecord := true

				for i, s := range rec {
					v, err := strconv.ParseFloat(s, 64)
					if err != nil {
						// If a value cannot be parsed, treat the entire record as invalid.
						fmt.Printf("Skipping Due to Parsing Error")
						validRecord = false
						break
					}

					if i == labelCol {
						y = v
					} else {
						x = append(x, v)
					}
				}
				if validRecord {
					out <- Sample{X: x, Y: y}
				}
			}
		}
	}()
	return done, nil
}

// Batch reads from a Sample channel and emits mini-batches via a channel
// Batch represents a collection of data points.
type Batch struct {
	X [][]float64
	Y []float64
}

func Batcher(in <-chan Sample, batchSize int, out chan<- Batch) (done chan struct{}) {
	// Create a channel that can be used to signal this goroutine to stop.
	done = make(chan struct{})

	// Start a new goroutine to run the batching logic concurrently.
	go func() {
		// Ensure the output channel is closed when the goroutine finishes.
		// This signals to receivers that no more batches will be sent.
		defer close(out)

		// Initialize slices to accumulate features (X) and labels (Y) for the current batch.
		var X [][]float64
		var Y []float64

		// Start an infinite loop to process samples from the input channel.
		for {
			// The select statement allows the goroutine to listen on multiple channels.
			select {
			case <-done:
				// If a signal is received on the 'done' channel,
				// it means the caller wants to stop processing early.
				// The goroutine exits gracefully.
				return

			case s, ok := <-in:
				// Receive a Sample from the input channel 'in'.
				// The 'ok' variable is false if the channel has been closed.
				if !ok {
					// The input channel is closed and no more samples are coming.
					// Check if there are any remaining samples in the current batch.
					if len(Y) > 0 {
						// Send the final, possibly incomplete, batch to the output channel.
						out <- Batch{X: X, Y: Y}
					}
					// Exit the goroutine.
					return
				}

				// Append the features and label from the received sample to our batch slices.
				X = append(X, s.X)
				Y = append(Y, s.Y)

				// Check if the current batch has reached the desired size.
				if len(Y) == batchSize {
					// If the batch is full, send it to the output channel.
					out <- Batch{X: X, Y: Y}
					// Reset the slices to start accumulating the next batch.
					// This is efficient as it reclaims the memory for the new batch.
					X = nil
					Y = nil
				}
			}
		}
	}()

	// Return the 'done' channel so the caller can signal to stop the process.
	return done
}
