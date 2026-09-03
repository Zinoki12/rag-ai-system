package backoff

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestPolicyDo(t *testing.T) {
	transient := Mark(errors.New("boom"), 0)
	permanent := errors.New("bad request")

	tests := []struct {
		name         string
		maxAttempts  int
		failures     []error // returned in order, then success
		wantAttempts int
		wantErr      bool
	}{
		{"успех сразу", 3, nil, 1, false},
		{"одна transient, потом успех", 3, []error{transient}, 2, false},
		{"исчерпали попытки", 3, []error{transient, transient, transient}, 3, true},
		{"постоянная ошибка не повторяется", 3, []error{permanent}, 1, true},
		{"MaxAttempts=1 запрещает повторы", 1, []error{transient}, 1, true},
		{"MaxAttempts=0 нормализуется до 1", 0, []error{transient}, 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := Policy{MaxAttempts: tt.maxAttempts, Base: time.Millisecond, Max: 2 * time.Millisecond}

			attempts := 0
			err := p.Do(context.Background(), func(context.Context) error {
				defer func() { attempts++ }()
				if attempts < len(tt.failures) {
					return tt.failures[attempts]
				}
				return nil
			})

			if (err != nil) != tt.wantErr {
				t.Errorf("Do() error = %v, wantErr = %v", err, tt.wantErr)
			}
			if attempts != tt.wantAttempts {
				t.Errorf("op ran %d time(s), want %d", attempts, tt.wantAttempts)
			}
		})
	}
}

// A permanent error must reach the caller intact — wrapping it in "giving up
// after N attempts" would be a lie, and errors.Is/As must keep working.
func TestPolicyDoPreservesPermanentError(t *testing.T) {
	sentinel := errors.New("bad api key")
	p := Policy{MaxAttempts: 5, Base: time.Millisecond}

	err := p.Do(context.Background(), func(context.Context) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Fatalf("Do() error = %v, want it to wrap %v", err, sentinel)
	}
}

func TestPolicyDoRespectsContext(t *testing.T) {
	t.Run("уже отменённый контекст не запускает операцию", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		called := false
		err := Policy{MaxAttempts: 3, Base: time.Millisecond}.Do(ctx, func(context.Context) error {
			called = true
			return nil
		})
		if err == nil {
			t.Error("Do() with a cancelled context returned no error")
		}
		if called {
			t.Error("Do() ran the operation despite a cancelled context")
		}
	})

	t.Run("ожидание бэкоффа прерывается отменой", func(t *testing.T) {
		// Base of an hour: only a ctx-aware sleep can finish this test.
		p := Policy{MaxAttempts: 5, Base: time.Hour, Max: time.Hour}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()

		start := time.Now()
		err := p.Do(ctx, func(context.Context) error { return Mark(errors.New("boom"), 0) })
		if err == nil {
			t.Fatal("Do() returned no error")
		}
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Errorf("Do() ignored the context and waited %v", elapsed)
		}
	})
}

// Retry-After must be honoured but never blindly: a server sending a huge value
// would otherwise stall the whole run.
func TestPolicyDelay(t *testing.T) {
	p := Policy{MaxAttempts: 5, Base: time.Second, Max: 10 * time.Second}

	tests := []struct {
		name    string
		attempt int
		hint    time.Duration
		wantMin time.Duration
		wantMax time.Duration
	}{
		{"подсказка сервера соблюдается", 1, 3 * time.Second, 3 * time.Second, 3 * time.Second},
		{"подсказка обрезается потолком", 1, time.Hour, 10 * time.Second, 10 * time.Second},
		{"без подсказки — джиттер в пределах окна", 1, 0, 0, time.Second},
		{"окно растёт экспоненциально", 3, 0, 0, 4 * time.Second},
		{"окно не превышает потолок", 10, 0, 0, 10 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for range 50 { // jitter is random; sample it
				got := p.delay(tt.attempt, tt.hint)
				if got < tt.wantMin || got > tt.wantMax {
					t.Fatalf("delay() = %v, want within [%v, %v]", got, tt.wantMin, tt.wantMax)
				}
			}
		})
	}
}

func TestIsTransientStatus(t *testing.T) {
	tests := []struct {
		code int
		want bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusForbidden, false},
		{http.StatusNotFound, false},
		{http.StatusRequestTimeout, true},
		{http.StatusTooManyRequests, true},
		{http.StatusInternalServerError, true},
		{http.StatusBadGateway, true},
		{http.StatusServiceUnavailable, true},
		{http.StatusGatewayTimeout, true},
	}

	for _, tt := range tests {
		if got := IsTransientStatus(tt.code); got != tt.want {
			t.Errorf("IsTransientStatus(%d) = %v, want %v", tt.code, got, tt.want)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	tests := []struct {
		name   string
		header string
		want   time.Duration
	}{
		{"пусто", "", 0},
		{"секунды", "30", 30 * time.Second},
		{"ноль секунд", "0", 0},
		{"мусор", "soon", 0},
		{"отрицательное", "-5", 0},
		{"дата в прошлом", "Mon, 02 Jan 2006 15:04:05 GMT", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseRetryAfter(tt.header); got != tt.want {
				t.Errorf("ParseRetryAfter(%q) = %v, want %v", tt.header, got, tt.want)
			}
		})
	}

	t.Run("дата в будущем", func(t *testing.T) {
		future := time.Now().Add(90 * time.Second).UTC().Format(http.TimeFormat)
		got := ParseRetryAfter(future)
		if got < 80*time.Second || got > 95*time.Second {
			t.Errorf("ParseRetryAfter(future) = %v, want roughly 90s", got)
		}
	})
}

func TestMarkTransport(t *testing.T) {
	t.Run("отмена контекста не ретраится", func(t *testing.T) {
		for _, err := range []error{context.Canceled, context.DeadlineExceeded} {
			var r *Retryable
			if errors.As(MarkTransport(err), &r) {
				t.Errorf("MarkTransport(%v) marked a context error as retryable", err)
			}
		}
	})

	t.Run("nil остаётся nil", func(t *testing.T) {
		if MarkTransport(nil) != nil {
			t.Error("MarkTransport(nil) returned non-nil")
		}
	})

	t.Run("не-сетевая ошибка не помечается", func(t *testing.T) {
		var r *Retryable
		if errors.As(MarkTransport(errors.New("json: bad token")), &r) {
			t.Error("MarkTransport marked a non-network error as retryable")
		}
	})
}
