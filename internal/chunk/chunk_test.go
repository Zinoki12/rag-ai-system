package chunk

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestCut(t *testing.T) {
	longParagraph := "ЭтоОченьДлинныйТекстДляПроверкиПринудительнойНарезкиСтрокБезПотериСимволов"

	tests := []struct {
		name         string
		text         string
		chunkSize    int
		wantCount    int
		wantExact    string // точное ожидаемое содержимое первого чанка (если нужно)
		wantErr      error
		validateJoin bool
	}{
		{
			name:      "Ошибка: chunkSize = 0",
			text:      "Какой-то текст",
			chunkSize: 0,
			wantErr:   ErrInvalidChunkSize,
		},
		{
			name:      "Ошибка: отрицательный chunkSize",
			text:      "Какой-то текст",
			chunkSize: -10,
			wantErr:   ErrInvalidChunkSize,
		},
		{
			name:      "Пустой текст",
			text:      "",
			chunkSize: 100,
			wantCount: 0,
			wantErr:   nil,
		},
		{
			name:      "Текст из одних пробелов и переводов строк",
			text:      "   \n \t \n   ",
			chunkSize: 100,
			wantCount: 0,
			wantErr:   nil,
		},
		{
			name:      "Один абзац короче лимита",
			text:      "Привет, мир!",
			chunkSize: 50,
			wantCount: 1,
			wantExact: "Привет, мир!",
			wantErr:   nil,
		},
		{
			name:      "Два коротких абзаца склеиваются в один с переносом",
			text:      "Первый абзац.\n\nВторой абзац.",
			chunkSize: 50,
			wantCount: 1,
			wantExact: "Первый абзац.\n\nВторой абзац.",
			wantErr:   nil,
		},
		{
			name:      "Два абзаца со строкой пробелов между ними (должен быть чистый разделитель \\n\\n)",
			text:      "Первый абзац.\n   \nВторой абзац.",
			chunkSize: 50,
			wantCount: 1,
			wantExact: "Первый абзац.\n\nВторой абзац.", // 👈 жесткая проверка содержимого
			wantErr:   nil,
		},
		{
			name:      "Два абзаца не влезают в один чанк",
			text:      "Первый длинный абзац.\n\nВторой длинный абзац.",
			chunkSize: 25,
			wantCount: 2,
			wantErr:   nil,
		},
		{
			name:         "Абзац длиннее лимита (принудительная нарезка по рунам)",
			text:         longParagraph,
			chunkSize:    15,
			wantCount:    5,
			wantErr:      nil,
			validateJoin: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Cut(tt.text, tt.chunkSize)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("expected error %v, got %v", tt.wantErr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(got) != tt.wantCount {
				t.Fatalf("got %d chunks, want %d, chunk: %s", len(got), tt.wantCount, got)
			}

			// Проверка точного совпадения текста (защита от склеивания мусора)
			if tt.wantExact != "" && len(got) > 0 {
				if got[0] != tt.wantExact {
					t.Errorf("chunk content mismatch.\nGot:  %q\nWant: %q", got[0], tt.wantExact)
				}
			}

			if tt.validateJoin {
				joined := strings.Join(got, "")
				if joined != tt.text {
					t.Errorf("joined chunks != original text.\nGot:  %q\nWant: %q", joined, tt.text)
				}
			}

			for i, ch := range got {
				if !utf8.ValidString(ch) {
					t.Errorf("chunk [%d] contains invalid UTF-8: %q", i, ch)
				}

				rCount := utf8.RuneCountInString(ch)
				if rCount > tt.chunkSize {
					t.Errorf("chunk [%d] rune count = %d, exceeds limit %d", i, rCount, tt.chunkSize)
				}
			}
		})
	}
}
