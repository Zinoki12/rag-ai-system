package vault

import (
	"errors"
	"testing"
)

func TestYamlParser(t *testing.T) {
	tests := []struct {
		name        string
		text        string
		wantTitle   string
		wantCreated string
		wantErr     error
	}{
		{
			name:        "Стандартный frontmatter",
			text:        "---\ntitle: pgvector и векторный поиск\ncreated: 2026-03-30\n---\nТело",
			wantTitle:   "pgvector и векторный поиск",
			wantCreated: "2026-03-30",
			wantErr:     nil,
		},
		{
			name:        "Заметка БЕЗ frontmatter",
			text:        "# Заголовок\nТекст без метаданных",
			wantTitle:   "",
			wantCreated: "",
			wantErr:     nil,
		},
		{
			name:        "Кавычки и двоеточия в заголовке",
			text:        "---\ntitle: \"RAG: поиск по базе\"\ncreated: '2026-01-01'\n---\nТекст",
			wantTitle:   "RAG: поиск по базе",
			wantCreated: "2026-01-01",
			wantErr:     nil,
		},
		{
			name:        "Незакрытый frontmatter (ждем ErrUnclosedFrontmatter)",
			text:        "---\ntitle: Битый файл без закрытия\ncreated: 2026-03-30\nТекст без закрывающих тире",
			wantTitle:   "",
			wantCreated: "",
			wantErr:     ErrUnclosedFrontmatter,
		},
		{
			name: "Разделитель --- посреди тела заметки",
			text: `---
title: Заметка с разделителем
---
Текст до черты
---
Текст после черты`,
			wantTitle:   "Заметка с разделителем",
			wantCreated: "",
			wantErr:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := yamlParser(tt.text)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("expected error %v, got %v", tt.wantErr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got.Title != tt.wantTitle {
				t.Errorf("got Title = %q, want %q", got.Title, tt.wantTitle)
			}
		})
	}
}
