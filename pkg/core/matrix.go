package core

import (
	"errors"
	"runtime"
	"sync"
)

type Matrix struct {
	R, C int
	Data []float64
}

// New Matrix Allocates Zero Matrix
func NewMatrix(r, c int) *Matrix {
	return &Matrix{R: r, C: c, Data: make([]float64, r*c)}
}

// FromSlice creates a Matrix from a nested slice (copies Matrix)
func FromSlice(a [][]float64) *Matrix {
	r := len(a)
	if r == 0 {
		return &Matrix{R: 0, C: 0}
	}

	c := len(a[0])
	m := NewMatrix(r, c)
	var k uint16 = 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Data[k] = a[i][j]
			k++
		}
	}
	return m
}

// At returns element (i, j)
func (m *Matrix) At(i, j int) float64 { return m.Data[i*m.C+j] }

// Set sets element (i, j)
func (m *Matrix) Set(i, j int) float64 { return m.Data[i*m.C+j] }

// Clone Deep Copies of Matrix
func (m *Matrix) Clone() *Matrix {
	n := &Matrix{R: m.R, C: m.C, Data: make([]float64, len(m.Data))}
	copy(n.Data, m.Data)
	return n
}

func (m *Matrix) Transpose() *Matrix {
	t := NewMatrix(m.C, m.R)
	for i := 0; i < m.R; i++ {
		for j := 0; j < m.C; j++ {
			t.Data[j*t.C+i] = m.Data[i*m.C+j]
		}
	}
	return t
}

// sync.Pool for temeperoray buffer to reduce GC pressure
var buffPool = sync.Pool{New: func() any { return make([]float64, 0, 1024) }}

// Dot computes dot(a, b) for vectors (as *Matrix with shape (n,1) or (1,n))
func Dot(a, b *Matrix) (float64, error) {
	if !(a.R == 1 || a.C == 1) || !(b.R == 1 || b.C == 1) {
		return 0, errors.New("Dot epxects Vectors")
	}

	n := a.R * a.C
	if n != b.R*b.C {
		return 0, errors.New("Mismatched Lengths")
	}

	sum := 0.0
	// parallel accumlation in chunks
	workers := runtime.GOMAXPROCS(0)
	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup
	var mu sync.Mutex

	for w := 0; w < workers; w++ {
		start := w * chunk
		end := min(start+chunk, n)
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			local := 0.0
			for i := s; i < e; i++ {
				local += a.Data[i] * b.Data[i]
			}
			mu.Lock()
			sum += local
			mu.Unlock()
		}(start, end)
	}
	wg.Wait()
	return sum, nil
}

func MatMul(A, B *Matrix) (*Matrix, error) {
	if A.C != B.R {
		return nil, errors.New("Dimension mismatch")
	}

	C := NewMatrix(A.R, B.C)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	rowsPerWorker := (A.R + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * rowsPerWorker
		end := min(start+rowsPerWorker, A.R)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(rs, re int) {
			defer wg.Done()
			for i := rs; i < re; i++ {
				for k := 0; k < A.C; k++ {
					ai := A.Data[i*A.C+k]
					for j := 0; j < B.C; j++ {
						C.Data[i*C.C+j] += ai * B.Data[k*B.C+j]
					}
				}
			}
		}(start, end)
	}
	wg.Wait()
	return C, nil
}

// Add returns A + B (element-wise).
func Add(A, B *Matrix) (*Matrix, error) {
	if A.R != B.R || A.C != B.C {
		return nil, errors.New("Dimension mismatch")
	}
	C := NewMatrix(A.R, A.C)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = A.Data[i] + B.Data[i]
	}
	return C, nil
}

// Sub returns A - B (element-wise).
func Sub(A, B *Matrix) (*Matrix, error) {
	if A.R != B.R || A.C != B.C {
		return nil, errors.New("Dimension mismatch")
	}
	C := NewMatrix(A.R, A.C)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = A.Data[i] - B.Data[i]
	}
	return C, nil
}

// Scale returns s*A.
func Scale(A *Matrix, s float64) *Matrix {
	C := NewMatrix(A.R, A.C)
	for i := 0; i < len(A.Data); i++ {
		C.Data[i] = s * A.Data[i]
	}
	return C
}

// Apply applies f element-wise (in-place, pointer receiver for efficiency).
func (m *Matrix) Apply(f func(float64) float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = f(m.Data[i])
	}
}

// RowSlice returns a view copy of row i as a column vector matrix (n,1).
func (m *Matrix) RowSlice(i int) *Matrix {
	v := NewMatrix(1, m.C)
	copy(v.Data, m.Data[i*m.C:(i+1)*m.C])
	return v
}

// ColSlice returns a view copy of column j as a row vector.
func (m *Matrix) ColSlice(j int) *Matrix {
	v := NewMatrix(m.R, 1)
	for i := 0; i < m.R; i++ {
		v.Data[i] = m.Data[i*m.C+j]
	}
	return v
}
