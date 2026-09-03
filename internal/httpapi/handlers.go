package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/Zinoki12/rag-ai-system/internal/app"
	"github.com/Zinoki12/rag-ai-system/internal/storage"
)

// maxRequestBody bounds how much a client may send. Questions are short; a body
// larger than this is a mistake or an attack, and either way reading it into
// memory is not the answer.
const maxRequestBody = 64 << 10

// Service is what the handlers need from the application.
//
// An interface rather than *app.App so the HTTP layer can be tested without a
// database, an embedding model or an API key — which is what makes it worth
// testing the error paths, the ones hardest to reach through a live stack.
type Service interface {
	Stats(ctx context.Context) (storage.EmbeddingStats, error)
	SpaceName() string
	Search(ctx context.Context, query string, topK int) ([]storage.Hit, error)
	Ask(ctx context.Context, question string, topK int) (app.Answer, error)
}

type errorResponse struct {
	Error     string `json:"error"`
	RequestID string `json:"request_id,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	// Encode before touching the header: a marshalling failure after
	// WriteHeader leaves a truncated body under a success status, which is
	// worse than an honest 500.
	body, err := json.Marshal(v)
	if err != nil {
		http.Error(w, `{"error":"failed to encode response"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func writeError(w http.ResponseWriter, r *http.Request, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg, RequestID: RequestID(r.Context())})
}

// decodeBody reads a JSON request body under a size limit, rejecting unknown
// fields so a typo in a client's field name fails loudly instead of silently
// falling back to a default.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBody)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	return nil
}

type healthResponse struct {
	Status string `json:"status"`
	Space  string `json:"space"`
	Chunks int    `json:"chunks"`
	Vector int    `json:"vectors"`
}

// handleHealth reports whether the service can actually serve, not merely
// whether the process is running: it touches the database and reports index
// coverage. A health check that cannot fail is not a health check.
func handleHealth(a Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		stats, err := a.Stats(r.Context())
		if err != nil {
			writeError(w, r, http.StatusServiceUnavailable, "database unavailable")
			return
		}
		writeJSON(w, http.StatusOK, healthResponse{
			Status: "ok",
			Space:  a.SpaceName(),
			Chunks: stats.Chunks,
			Vector: stats.Embedded,
		})
	}
}

type searchRequest struct {
	Query string `json:"query"`
	K     int    `json:"k"`
}

type hitResponse struct {
	Score      float64 `json:"score"`
	NotePath   string  `json:"note_path"`
	NoteName   string  `json:"note_name,omitempty"`
	ChunkIndex int     `json:"chunk_index"`
	Text       string  `json:"text"`
}

type searchResponse struct {
	Space string        `json:"space"`
	Hits  []hitResponse `json:"hits"`
}

func handleSearch(a Service, log *slog.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req searchRequest
		if err := decodeBody(w, r, &req); err != nil {
			writeError(w, r, http.StatusBadRequest, "invalid request body: "+err.Error())
			return
		}
		query := strings.TrimSpace(req.Query)
		if query == "" {
			writeError(w, r, http.StatusBadRequest, "query is required")
			return
		}

		hits, err := a.Search(r.Context(), query, clampK(req.K))
		if err != nil {
			respondUpstreamError(w, r, log, "search failed", err)
			return
		}
		writeJSON(w, http.StatusOK, searchResponse{Space: a.SpaceName(), Hits: toHits(hits)})
	}
}

type askRequest struct {
	Question string `json:"question"`
	K        int    `json:"k"`
}

type askResponse struct {
	Answer  string   `json:"answer"`
	Model   string   `json:"model"`
	Space   string   `json:"space"`
	Sources []string `json:"sources"`
}

func handleAsk(a Service, log *slog.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req askRequest
		if err := decodeBody(w, r, &req); err != nil {
			writeError(w, r, http.StatusBadRequest, "invalid request body: "+err.Error())
			return
		}
		question := strings.TrimSpace(req.Question)
		if question == "" {
			writeError(w, r, http.StatusBadRequest, "question is required")
			return
		}

		answer, err := a.Ask(r.Context(), question, clampK(req.K))
		if err != nil {
			respondUpstreamError(w, r, log, "ask failed", err)
			return
		}
		writeJSON(w, http.StatusOK, askResponse{
			Answer:  answer.Text,
			Model:   answer.Model,
			Space:   a.SpaceName(),
			Sources: answer.Sources,
		})
	}
}

// respondUpstreamError logs the real cause and tells the client only what it
// can act on. Upstream errors carry API keys, hostnames and quota details that
// have no business in an HTTP response.
func respondUpstreamError(w http.ResponseWriter, r *http.Request, log *slog.Logger, msg string, err error) {
	if errors.Is(err, r.Context().Err()) && r.Context().Err() != nil {
		writeError(w, r, http.StatusGatewayTimeout, "request timed out")
		return
	}
	log.LogAttrs(r.Context(), slog.LevelError, msg,
		slog.String("request_id", RequestID(r.Context())),
		slog.String("error", err.Error()),
	)
	writeError(w, r, http.StatusBadGateway, msg)
}

// clampK keeps top-k inside sane bounds: zero means "use the default", and an
// unbounded value would let one request pull the whole table into a prompt.
func clampK(k int) int {
	const (
		defaultK = 5
		maxK     = 25
	)
	switch {
	case k <= 0:
		return defaultK
	case k > maxK:
		return maxK
	default:
		return k
	}
}

func toHits(hits []storage.Hit) []hitResponse {
	// Non-nil so an empty result encodes as [] rather than null.
	out := make([]hitResponse, 0, len(hits))
	for _, h := range hits {
		out = append(out, hitResponse{
			Score:      h.Score,
			NotePath:   h.NotePath,
			NoteName:   h.NoteName,
			ChunkIndex: h.ChunkIndex,
			Text:       h.Text,
		})
	}
	return out
}
