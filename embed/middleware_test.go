package embed

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/backoff"
)

// stubProvider records what it was asked and replays a scripted sequence of
// results, one per call.
type stubProvider struct {
	space    Space
	maxBatch int
	calls    [][]string
	kinds    []Kind
	errs     []error // errs[i] is returned by call i; nil means success
}

func (s *stubProvider) Space() Space  { return s.space }
func (s *stubProvider) MaxBatch() int { return s.maxBatch }

func (s *stubProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	n := len(s.calls)
	s.calls = append(s.calls, append([]string(nil), texts...))
	s.kinds = append(s.kinds, kind)

	if n < len(s.errs) && s.errs[n] != nil {
		return nil, s.errs[n]
	}
	out := make([]Vector, len(texts))
	for i, t := range texts {
		v := make(Vector, s.space.Dim)
		v[0] = float32(len(t)) // enough to tell vectors apart
		out[i] = v
	}
	return out, nil
}

func newStub(maxBatch int, errs ...error) *stubProvider {
	return &stubProvider{
		space:    Space{Provider: "stub", Model: "m", Dim: 4},
		maxBatch: maxBatch,
		errs:     errs,
	}
}

func TestBatched(t *testing.T) {
	tests := []struct {
		name      string
		maxBatch  int
		texts     []string
		wantCalls [][]string
	}{
		{
			name:      "пустой вход не трогает провайдера",
			maxBatch:  3,
			texts:     nil,
			wantCalls: nil,
		},
		{
			name:      "один текст — один вызов",
			maxBatch:  3,
			texts:     []string{"a"},
			wantCalls: [][]string{{"a"}},
		},
		{
			name:      "ровно предел батча — один вызов",
			maxBatch:  3,
			texts:     []string{"a", "b", "c"},
			wantCalls: [][]string{{"a", "b", "c"}},
		},
		{
			name:      "предел плюс один — два вызова, хвост отдельно",
			maxBatch:  3,
			texts:     []string{"a", "b", "c", "d"},
			wantCalls: [][]string{{"a", "b", "c"}, {"d"}},
		},
		{
			name:      "нулевой MaxBatch не делит на ноль",
			maxBatch:  0,
			texts:     []string{"a", "b"},
			wantCalls: [][]string{{"a"}, {"b"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub := newStub(tt.maxBatch)
			got, err := Batched(context.Background(), stub, tt.texts, KindDocument)
			if err != nil {
				t.Fatalf("Batched() error = %v", err)
			}
			if len(got) != len(tt.texts) {
				t.Errorf("got %d vectors for %d texts", len(got), len(tt.texts))
			}
			if fmt.Sprint(stub.calls) != fmt.Sprint(tt.wantCalls) {
				t.Errorf("provider calls = %v, want %v", stub.calls, tt.wantCalls)
			}
		})
	}
}

// A batch boundary is where a reordering bug would hide: each call returns a
// correct slice, and only the concatenation is wrong.
func TestBatchedPreservesOrderAcrossBatches(t *testing.T) {
	stub := newStub(2)
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}

	got, err := Batched(context.Background(), stub, texts, KindDocument)
	if err != nil {
		t.Fatalf("Batched() error = %v", err)
	}
	for i, text := range texts {
		if want := float32(len(text)); got[i][0] != want {
			t.Errorf("vector %d = %v, want marker %v (order scrambled)", i, got[i][0], want)
		}
	}
}

func TestBatchedRejectsShortBatch(t *testing.T) {
	// A provider that drops an input would otherwise shift every later vector
	// onto the wrong chunk.
	stub := &stubProvider{space: Space{Provider: "stub", Model: "m", Dim: 4}, maxBatch: 4}
	dropping := &droppingProvider{stubProvider: stub}

	if _, err := Batched(context.Background(), dropping, []string{"a", "b"}, KindDocument); err == nil {
		t.Fatal("Batched() accepted a batch with a missing vector")
	}
}

type droppingProvider struct{ *stubProvider }

func (d *droppingProvider) Embed(ctx context.Context, texts []string, kind Kind) ([]Vector, error) {
	out, err := d.stubProvider.Embed(ctx, texts, kind)
	if err != nil || len(out) == 0 {
		return out, err
	}
	return out[:len(out)-1], nil
}

func TestBatchedPassesKindThrough(t *testing.T) {
	stub := newStub(1)
	if _, err := Batched(context.Background(), stub, []string{"a", "b"}, KindQuery); err != nil {
		t.Fatalf("Batched() error = %v", err)
	}
	for i, k := range stub.kinds {
		if k != KindQuery {
			t.Errorf("call %d got kind %v, want %v", i, k, KindQuery)
		}
	}
}

func TestWithRetry(t *testing.T) {
	transient := backoff.Mark(errors.New("503 upstream"), 0)
	permanent := errors.New("401 bad api key")
	policy := backoff.Policy{MaxAttempts: 3, Base: time.Millisecond, Max: 2 * time.Millisecond}

	tests := []struct {
		name      string
		errs      []error
		wantCalls int
		wantErr   bool
	}{
		{"успех с первого раза", nil, 1, false},
		{"transient, потом успех", []error{transient}, 2, false},
		{"два transient, потом успех", []error{transient, transient}, 3, false},
		{"transient до исчерпания попыток", []error{transient, transient, transient}, 3, true},
		{"неретраибельная ошибка не повторяется", []error{permanent}, 1, true},
		{"неретраибельная после transient обрывает цикл", []error{transient, permanent}, 2, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub := newStub(10, tt.errs...)
			_, err := WithRetry(stub, policy).Embed(context.Background(), []string{"a"}, KindDocument)

			if (err != nil) != tt.wantErr {
				t.Errorf("Embed() error = %v, wantErr = %v", err, tt.wantErr)
			}
			if len(stub.calls) != tt.wantCalls {
				t.Errorf("provider called %d time(s), want %d", len(stub.calls), tt.wantCalls)
			}
		})
	}
}

func TestWithRetryStopsOnCancelledContext(t *testing.T) {
	stub := newStub(10, backoff.Mark(errors.New("503"), time.Hour))
	policy := backoff.Policy{MaxAttempts: 5, Base: time.Hour, Max: time.Hour}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	if _, err := WithRetry(stub, policy).Embed(ctx, []string{"a"}, KindDocument); err == nil {
		t.Fatal("Embed() returned no error after the context expired")
	}
	// Without a ctx-aware sleep this would block for an hour.
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("backoff ignored the context: waited %v", elapsed)
	}
}

func TestWithRateLimit(t *testing.T) {
	t.Run("нулевой rps отключает лимитер", func(t *testing.T) {
		stub := newStub(10)
		if p := WithRateLimit(stub, 0, 1); p != Provider(stub) {
			t.Error("WithRateLimit(rps=0) wrapped the provider instead of returning it unchanged")
		}
	})

	t.Run("отменённый контекст возвращает ошибку, не вызывая провайдера", func(t *testing.T) {
		stub := newStub(10)
		p := WithRateLimit(stub, 0.001, 1)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		if _, err := p.Embed(ctx, []string{"a"}, KindDocument); err == nil {
			t.Fatal("Embed() with a cancelled context returned no error")
		}
		if len(stub.calls) != 0 {
			t.Errorf("provider was called %d time(s) despite a cancelled context", len(stub.calls))
		}
	})

	t.Run("лимитер держит скорость", func(t *testing.T) {
		stub := newStub(10)
		p := WithRateLimit(stub, 100, 1) // 10ms between calls, burst of 1

		start := time.Now()
		for range 3 {
			if _, err := p.Embed(context.Background(), []string{"a"}, KindDocument); err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
		}
		// First call is free (burst), the next two wait ~10ms each.
		if elapsed := time.Since(start); elapsed < 15*time.Millisecond {
			t.Errorf("3 calls at 100 rps took %v, want at least ~20ms", elapsed)
		}
	})
}
