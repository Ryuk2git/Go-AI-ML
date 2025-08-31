package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"aiml/pkg/dataprep"
	"aiml/pkg/stats"
)

//
// ---------------------- CLI FLAGS DOCUMENTATION ----------------------
//
// --input         : Path to input CSV file. Default = Employee.csv
// --mode          : Output mode: "cli" (preview in console) or "csv" (save processed file)
// --output        : Path to save processed CSV (if mode=csv). Default = ./processed_<input>
// --preview       : Number of rows to preview in console
// --label-col     : Index of label column (0-based). Use -1 if no labels
// --missing-thresh: Drop columns with > threshold fraction missing values. Default=0.2
// --poly-degree   : Polynomial feature expansion degree. (1 = no expansion)
// --encode        : Encoding method for categorical vars: "none", "label", "onehot", "freq", "auto"
// --auto          : If true, runs full pipeline (missing, encoding, scaling, poly, output CSV)
//
// Example:
//   go run main.go --input Employee.csv --mode csv --label-col 8 --encode auto --poly-degree 2 --auto
//
// ---------------------------------------------------------------------
//

// previewData prints the first N rows of float64 slices with headers
func previewData(headers []string, X [][]float64, Y []float64, n int, labelCol bool) {
	if n > len(X) {
		n = len(X)
	}

	// Print headers
	for _, h := range headers {
		fmt.Printf("%-15s", h)
	}
	if labelCol {
		fmt.Printf("%-15s", "Label")
	}
	fmt.Println()

	// Print rows
	for i := 0; i < n; i++ {
		for _, val := range X[i] {
			fmt.Printf("%-15.6f", val)
		}
		if labelCol {
			fmt.Printf("%-15.6f", Y[i])
		}
		fmt.Println()
	}
}

func main() {
	// ---- CLI Flags ----
	inputPath := flag.String("input", "Employee.csv", "Path to input CSV file")
	mode := flag.String("mode", "cli", "Output mode: cli or csv")
	outputPath := flag.String("output", "", "Path to save processed CSV (if mode=csv)")
	previewRows := flag.Int("preview", 5, "Number of rows to preview in console")
	labelCol := flag.Int("label-col", -1, "Index of label column (-1 if no labels)")
	missingThresh := flag.Float64("missing-thresh", 0.2, "Threshold for dropping columns with too many missing values")
	polyDegree := flag.Int("poly-degree", 1, "Degree for polynomial features (1 = no expansion)")
	encodeMethod := flag.String("encode", "none", "Encoding: none, label, onehot, freq, auto")
	auto := flag.Bool("auto", false, "Run full pipeline automatically and save CSV")
	flag.Parse()

	// ---- Load raw CSV ----
	file, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("Error opening CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rawData, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error reading CSV file: %v", err)
	}

	if len(rawData) < 2 {
		log.Fatalf("CSV file has no data rows")
	}

	fmt.Printf("Loaded raw data: %d rows, %d columns\n", len(rawData)-1, len(rawData[0]))

	// ---- Headers and Data ----
	headers := rawData[0] // original headers
	dataRows := rawData[1:]

	// ---- Handle Missing Values ----
	cleaned := dataprep.HandleMissingValues(append([][]string{headers}, dataRows...), *missingThresh)
	headers = cleaned[0]
	dataRows = cleaned[1:]

	// ---- Encoding ----
	var X [][]float64
	var encoders map[string]interface{}
	var featureHeaders []string

	if *encodeMethod != "none" && *encodeMethod != "" {
		X, encoders = dataprep.EncodeCategoricalAll(dataRows, headers, *encodeMethod)

		// Generate feature headers
		for _, h := range headers {
			if e, ok := encoders[h]; ok {
				// One-hot expands to multiple columns
				if mapping, ok := e.(map[string]int); ok && len(mapping) > 1 {
					for cat := range mapping {
						featureHeaders = append(featureHeaders, h+"_"+cat)
					}
				} else {
					featureHeaders = append(featureHeaders, h)
				}
			} else {
				featureHeaders = append(featureHeaders, h)
			}
		}
	} else {
		// Convert dataRows ([][]string) to [][]float64, skipping label column if present
		X = make([][]float64, len(dataRows))
		for i, row := range dataRows {
			var floatRow []float64
			for j, val := range row {
				if *labelCol >= 0 && j == *labelCol {
					continue // skip label column
				}
				f, err := strconv.ParseFloat(val, 64)
				if err != nil {
					f = 0
				}
				floatRow = append(floatRow, f)
			}
			X[i] = floatRow
		}
		featureHeaders = headers
	}

	// ---- Handle Label Column ----
	var Y []float64
	if *labelCol >= 0 {
		for i := 0; i < len(X); i++ {
			val, err := strconv.ParseFloat(dataRows[i][*labelCol], 64)
			if err != nil {
				val = 0
			}
			Y = append(Y, val)
			// Remove label column from X
			X[i] = append(X[i][:*labelCol], X[i][*labelCol+1:]...)
		}
		featureHeaders = append(featureHeaders[:*labelCol], featureHeaders[*labelCol+1:]...)
	}

	fmt.Printf("After preprocessing: %d samples, %d features\n", len(X), len(X[0]))

	// ---- Drop Duplicates ----
	X = dataprep.DropDuplicates(X)

	// ---- Scaling ----
	X = stats.StandardizeData(X)

	// ---- Polynomial Features ----
	if *polyDegree > 1 {
		X, featureHeaders = dataprep.PolynomialFeaturesWithNames(X, featureHeaders, *polyDegree)
	}

	// ---- Output ----
	if *mode == "csv" || *auto {
		if *outputPath == "" {
			base := filepath.Base(*inputPath)
			*outputPath = filepath.Join(".", "processed_"+base)
		}

		file, err := os.Create(*outputPath)
		if err != nil {
			log.Fatalf("Error creating output file: %v", err)
		}
		defer file.Close()

		writer := csv.NewWriter(file)
		defer writer.Flush()

		// Write headers
		outHeaders := featureHeaders
		if *labelCol >= 0 {
			outHeaders = append(outHeaders, headers[*labelCol])
		}
		if err := writer.Write(outHeaders); err != nil {
			log.Fatalf("Error writing headers: %v", err)
		}

		// Write rows
		for i, row := range X {
			stringRow := make([]string, len(row))
			for j, val := range row {
				stringRow[j] = strconv.FormatFloat(val, 'f', 6, 64)
			}
			if *labelCol >= 0 {
				stringRow = append(stringRow, strconv.FormatFloat(Y[i], 'f', 6, 64))
			}
			if err := writer.Write(stringRow); err != nil {
				log.Fatalf("Error writing row: %v", err)
			}
		}

		fmt.Println("Processed data saved to:", *outputPath)
	} else {
		fmt.Println("\nPreview of processed data:")
		previewData(featureHeaders, X, Y, *previewRows, *labelCol >= 0)
	}
}
