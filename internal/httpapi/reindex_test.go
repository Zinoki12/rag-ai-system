package httpapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/Zinoki12/rag-ai-system/knowledge"
)

func decodeStatus(t *testing.T, body []byte) indexStatus {
	t.Helper()
	var st indexStatus
	if err := json.Unmarshal(body, &st); err != nil {
		t.Fatalf("decode status: %v (body: %s)", err, body)
	}
	return st
}

// waitForState polls the status endpoint until it leaves "running".
func waitForState(t *testing.T, h http.Handler, want string) indexStatus {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		st := decodeStatus(t, do(t, h, http.MethodGet, "/reindex/status", "").Body.Bytes())
		if st.State == want {
			return st
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("status never reached %q", want)
	return indexStatus{}
}

func TestReindexStartsAndReports(t *testing.T) {
	svc := &stubService{indexRes: knowledge.IndexResult{
		NotesWritten: 3, NotesUnchanged: 2, NotesDeleted: 1, ChunksEmbedded: 12,
	}}
	h := newTestHandler(svc)

	// 202, not 200: the work has been accepted, not finished.
	w := do(t, h, http.MethodPost, "/reindex", "")
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202 (body: %s)", w.Code, w.Body.String())
	}

	st := waitForState(t, h, stateDone)
	if st.Result == nil {
		t.Fatal("finished status carries no result")
	}
	if st.Result.NotesWritten != 3 || st.Result.ChunksEmbedded != 12 {
		t.Errorf("result = %+v, want the stub's counts", *st.Result)
	}
	if st.StartedAt == nil || st.FinishedAt == nil {
		t.Error("status is missing timestamps")
	}
}

func TestReindexStatusStartsIdle(t *testing.T) {
	st := decodeStatus(t, do(t, newTestHandler(&stubService{}), http.MethodGet, "/reindex/status", "").Body.Bytes())
	if st.State != stateIdle {
		t.Errorf("state = %q, want %q before any pass", st.State, stateIdle)
	}
}

// Two passes over one vault at the same time would embed the same chunks twice
// and race on the same rows; the second caller is told one is already running.
func TestReindexRefusesConcurrentPasses(t *testing.T) {
	svc := &stubService{indexHold: make(chan struct{})}
	h := newTestHandler(svc)

	if w := do(t, h, http.MethodPost, "/reindex", ""); w.Code != http.StatusAccepted {
		t.Fatalf("first request status = %d, want 202", w.Code)
	}
	waitForState(t, h, stateRunning)

	w := do(t, h, http.MethodPost, "/reindex", "")
	if w.Code != http.StatusConflict {
		t.Errorf("second request status = %d, want 409", w.Code)
	}
	if st := decodeStatus(t, w.Body.Bytes()); st.State != stateRunning {
		t.Errorf("conflict body state = %q, want %q", st.State, stateRunning)
	}

	close(svc.indexHold)
	waitForState(t, h, stateDone)

	if n := svc.indexRuns.Load(); n != 1 {
		t.Errorf("Index ran %d times, want exactly 1", n)
	}
}

// The pass must outlive the request that started it. Handing it r.Context()
// would cancel it a millisecond after the handler returned 202.
func TestReindexOutlivesTheRequest(t *testing.T) {
	svc := &stubService{indexHold: make(chan struct{})}
	h := newTestHandler(svc)

	do(t, h, http.MethodPost, "/reindex", "")
	waitForState(t, h, stateRunning)

	// The request has long since returned; the pass is still going.
	time.Sleep(50 * time.Millisecond)
	if st := waitForState(t, h, stateRunning); st.State != stateRunning {
		t.Fatal("the pass did not survive the request that started it")
	}

	close(svc.indexHold)
	st := waitForState(t, h, stateDone)
	if st.Error != "" {
		t.Errorf("pass failed after the request returned: %s", st.Error)
	}
}

// Index reports partial failures alongside a usable result, so the counts must
// survive onto the failure path — otherwise an operator sees only "failed" and
// cannot tell whether anything was indexed at all.
func TestReindexFailureKeepsCounts(t *testing.T) {
	svc := &stubService{
		indexRes: knowledge.IndexResult{NotesWritten: 2, NotesFailed: 1},
		indexErr: errors.New("note broken.md: value too long"),
	}
	h := newTestHandler(svc)

	do(t, h, http.MethodPost, "/reindex", "")
	st := waitForState(t, h, stateFailed)

	if st.Error == "" {
		t.Error("failed status carries no error message")
	}
	if st.Result == nil || st.Result.NotesWritten != 2 || st.Result.NotesFailed != 1 {
		t.Errorf("result = %+v, want the partial counts preserved", st.Result)
	}
}

func TestReindexRejectsWrongMethod(t *testing.T) {
	h := newTestHandler(&stubService{})
	if w := do(t, h, http.MethodGet, "/reindex", ""); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /reindex = %d, want 405", w.Code)
	}
	if w := do(t, h, http.MethodPost, "/reindex/status", ""); w.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST /reindex/status = %d, want 405", w.Code)
	}
}
