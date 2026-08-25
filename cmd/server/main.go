package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

type HealthResponse struct {
	Status string `json:"status"`
}

func HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		response := HealthResponse{
			Status: "ok",
		}
		w.Header().Set("Content-type", "application/json")

		var buf bytes.Buffer
		if err := json.NewEncoder(&buf).Encode(response); err != nil {
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(buf.Bytes())
	}
}

func ServerStart(mux *http.ServeMux) error {
	mux.HandleFunc("/health", HealthHandler())

	log.Println("Сервер запускается...")

	err := http.ListenAndServe(":8081", mux)
	if err != nil {
		return fmt.Errorf("не удалось запустить сервер %w", err)
	}
	return nil
}

func main() {
	mux := http.NewServeMux()

	err := ServerStart(mux)
	if err != nil {
		log.Fatalf("Критическая ошибка запуска %v", err)
	}
}
