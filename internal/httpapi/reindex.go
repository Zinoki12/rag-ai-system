package httpapi

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

// Indexing states reported by GET /reindex/status.
const (
	stateIdle    = "idle"
	stateRunning = "running"
	stateDone    = "done"
	stateFailed  = "failed"
)

// indexStatus is a snapshot of the last or current indexing pass.
type indexStatus struct {
	State      string                 `json:"state"`
	StartedAt  *time.Time             `json:"started_at,omitempty"`
	FinishedAt *time.Time             `json:"finished_at,omitempty"`
	Result     *knowledge.IndexResult `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

// indexer runs indexing passes in the background, one at a time.
//
// Indexing is not a request-shaped operation: it walks a directory and may call
// an embedding provider hundreds of times, so a synchronous endpoint would hold
// a connection open for the whole pass and time out on any real vault. The
// caller gets 202 and polls the status instead.
type indexer struct {
	svc Service
	log *slog.Logger

	// base outlives any single request. A background job started from a
	// request must not inherit that request's context, which is cancelled the
	// moment the handler returns — the pass would die a millisecond after it
	// began.
	base context.Context

	mu     sync.Mutex
	status indexStatus
}

func newIndexer(base context.Context, svc Service, log *slog.Logger) *indexer {
	return &indexer{svc: svc, log: log, base: base, status: indexStatus{State: stateIdle}}
}

// start begins a pass. It reports false when one is already running.
func (ix *indexer) start() bool {
	ix.mu.Lock()
	if ix.status.State == stateRunning {
		ix.mu.Unlock()
		return false
	}
	now := time.Now()
	ix.status = indexStatus{State: stateRunning, StartedAt: &now}
	ix.mu.Unlock()

	go ix.run()
	return true
}

func (ix *indexer) run() {
	res, err := ix.svc.Index(ix.base)
	finished := time.Now()

	ix.mu.Lock()
	defer ix.mu.Unlock()

	ix.status.FinishedAt = &finished
	ix.status.Result = &res
	if err != nil {
		// Index reports partial failures alongside a usable result, so the
		// counts are kept even on the failure path.
		ix.status.State = stateFailed
		ix.status.Error = err.Error()
		ix.log.Error("indexing failed", slog.String("error", err.Error()))
		return
	}
	ix.status.State = stateDone
	ix.log.Info("indexing finished",
		slog.Int("written", res.NotesWritten),
		slog.Int("unchanged", res.NotesUnchanged),
		slog.Int("deleted", res.NotesDeleted),
		slog.Int("embedded", res.ChunksEmbedded),
		slog.Duration("took", finished.Sub(*ix.status.StartedAt)),
	)
}

// snapshot returns a copy safe to serialise outside the lock.
func (ix *indexer) snapshot() indexStatus {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return ix.status
}
