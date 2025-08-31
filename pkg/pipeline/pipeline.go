package pipeline

// Transformer interface for fit/transform pattern.
type Transformer interface {
	Fit(X [][]float64, Y []float64)
	Transform(X [][]float64) [][]float64
}

// Pipeline chains multiple transformers.
type Pipeline struct {
	steps []Transformer
}

func NewPipeline(steps ...Transformer) *Pipeline {
	return &Pipeline{steps: steps}
}

func (p *Pipeline) Fit(X [][]float64, Y []float64) {
	for _, step := range p.steps {
		step.Fit(X, Y)
		X = step.Transform(X)
	}
}

func (p *Pipeline) Transform(X [][]float64) [][]float64 {
	for _, step := range p.steps {
		X = step.Transform(X)
	}
	return X
}
