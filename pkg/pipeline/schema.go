package pipeline

// Schema describes the structure of a dataset.
type Schema struct {
	FeatureNames []string
	Types        []string // e.g., "float", "int", "category"
}
