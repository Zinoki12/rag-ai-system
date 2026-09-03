package rag

import (
	"strings"
	"testing"
)

func passages(n int, textLen int) []Passage {
	out := make([]Passage, n)
	for i := range out {
		out[i] = Passage{
			Source: "note.md",
			Index:  i,
			Text:   strings.Repeat("я", textLen),
		}
	}
	return out
}

func TestBuildUserPrompt(t *testing.T) {
	t.Run("фрагменты нумеруются с единицы", func(t *testing.T) {
		got := BuildUserPrompt("вопрос", []Passage{
			{Source: "a.md", Index: 0, Text: "первый"},
			{Source: "b.md", Index: 3, Text: "второй"},
		}, 0)

		for _, want := range []string{"[1] a.md #0", "[2] b.md #3", "первый", "второй", "Вопрос: вопрос"} {
			if !strings.Contains(got, want) {
				t.Errorf("prompt is missing %q:\n%s", want, got)
			}
		}
	})

	t.Run("заголовок заметки попадает в шапку фрагмента", func(t *testing.T) {
		got := BuildUserPrompt("q", []Passage{
			{Source: "rag/chunking.md", Title: "Стратегии чанкинга", Index: 2, Text: "текст"},
		}, 0)

		if !strings.Contains(got, "Стратегии чанкинга — rag/chunking.md #2") {
			t.Errorf("prompt header is wrong:\n%s", got)
		}
	})

	t.Run("вопрос обрезается по краям", func(t *testing.T) {
		got := BuildUserPrompt("  что такое чанкинг  ", nil, 0)
		if !strings.HasSuffix(got, "Вопрос: что такое чанкинг") {
			t.Errorf("question was not trimmed:\n%q", got)
		}
	})

	t.Run("пустая выдача honest про отсутствие находок", func(t *testing.T) {
		got := BuildUserPrompt("вопрос", nil, 0)
		if !strings.Contains(got, "ничего не найдено") {
			t.Errorf("prompt does not say the search was empty:\n%s", got)
		}
	})
}

// Without a budget a large top-k silently overflows the model's context window:
// hosted APIs reject the call and local models truncate from the front, which
// drops the system instruction first.
func TestBuildUserPromptRespectsBudget(t *testing.T) {
	tests := []struct {
		name         string
		passages     []Passage
		budget       int
		wantIncluded int
	}{
		{"всё влезает", passages(3, 100), 1000, 3},
		{"хвост отбрасывается", passages(10, 100), 350, 3},
		{"первый фрагмент включается, даже если сам больше бюджета", passages(3, 5000), 100, 1},
		{"нулевой бюджет означает умолчание", passages(3, 100), 0, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildUserPrompt("вопрос", tt.passages, tt.budget)
			if n := strings.Count(got, "note.md #"); n != tt.wantIncluded {
				t.Errorf("included %d passages, want %d", n, tt.wantIncluded)
			}
		})
	}
}

// Truncation must drop the least relevant material. Search returns best first,
// so it has to cut from the end.
func TestBuildUserPromptKeepsBestMatchesFirst(t *testing.T) {
	got := BuildUserPrompt("вопрос", []Passage{
		{Source: "best.md", Index: 0, Text: strings.Repeat("а", 200)},
		{Source: "worst.md", Index: 0, Text: strings.Repeat("б", 200)},
	}, 250)

	if !strings.Contains(got, "best.md") {
		t.Error("the top match was dropped")
	}
	if strings.Contains(got, "worst.md") {
		t.Error("the weakest match survived the budget")
	}
}

func TestSources(t *testing.T) {
	tests := []struct {
		name     string
		passages []Passage
		want     []string
	}{
		{"пусто", nil, nil},
		{
			name: "несколько чанков одной заметки — один источник",
			passages: []Passage{
				{Source: "a.md", Index: 0},
				{Source: "a.md", Index: 1},
				{Source: "a.md", Index: 5},
			},
			want: []string{"a.md"},
		},
		{
			name: "порядок первого появления сохраняется",
			passages: []Passage{
				{Source: "b.md", Index: 0},
				{Source: "a.md", Index: 0},
				{Source: "b.md", Index: 1},
			},
			want: []string{"b.md", "a.md"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sources(tt.passages)
			if len(got) != len(tt.want) {
				t.Fatalf("Sources() = %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Sources()[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}
