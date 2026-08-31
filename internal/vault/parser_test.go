package vault

import (
	"errors"
	"testing"
)

func TestYamlParser(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantTitle string
		wantBody  string
		wantErr   error
	}{
		{
			name: "Стандартный frontmatter и тело статьи",
			input: `---
title: Введение в pgvector
---
# Заголовок
Это основной текст статьи.`,
			wantTitle: "Введение в pgvector",
			wantBody:  "# Заголовок\nЭто основной текст статьи.",
			wantErr:   nil,
		},
		{
			name: "Текст БЕЗ frontmatter (весь файл становится телом)",
			input: `# Чистый Markdown
Без всяких YAML заголовков.`,
			wantTitle: "",
			wantBody:  "# Чистый Markdown\nБез всяких YAML заголовков.",
			wantErr:   nil,
		},
		{
			name:      "Файл со скрытым UTF-8 BOM в начале",
			input:     "\ufeff---\ntitle: BOM Test\n---\nТекст после BOM",
			wantTitle: "BOM Test",
			wantBody:  "Текст после BOM",
			wantErr:   nil,
		},
		{
			name: "Пустые строки перед открывающим ---",
			input: `

---
title: С отступом
---
Тело заметки`,
			wantTitle: "С отступом",
			wantBody:  "Тело заметки",
			wantErr:   nil,
		},
		{
			name: "Кавычки и двоеточия в заголовке",
			input: `---
title: "RAG: поиск по базе данных: часть 1"
---
Контент статьи`,
			wantTitle: "RAG: поиск по базе данных: часть 1",
			wantBody:  "Контент статьи",
			wantErr:   nil,
		},
		{
			name: "Разделитель --- (горизонтальная черта) внутри тела статьи",
			input: `---
title: Заметка с разделителем
---
Текст до черты
---
Текст после черты`,
			wantTitle: "Заметка с разделителем",
			wantBody:  "Текст до черты\n---\nТекст после черты",
			wantErr:   nil,
		},
		{
			name: "Сохранение отступов блоков кода в теле статьи",
			input: `---
title: Code Snippet
---
Пример кода:
    func main() {
        fmt.Println("hi")
    }`,
			wantTitle: "Code Snippet",
			wantBody:  "Пример кода:\n    func main() {\n        fmt.Println(\"hi\")\n    }",
			wantErr:   nil,
		},
		{
			name: "Разный регистр ключа (TITLE:)",
			input: `---
TITLE: Caps Title
---
Тело`,
			wantTitle: "Caps Title",
			wantBody:  "Тело",
			wantErr:   nil,
		},
		{
			name: "Ошибка: незакрытый frontmatter (ErrUnclosedFrontmatter)",
			input: `---
title: Битый файл
Текст пошел без закрывающих трех тире`,
			wantTitle: "",
			wantBody:  "",
			wantErr:   ErrUnclosedFrontmatter,
		},
		{
			name:      "Пустая строка на входе",
			input:     "",
			wantTitle: "",
			wantBody:  "",
			wantErr:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := yamlParser(tt.input)

			// 1. Проверяем ошибки через errors.Is
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("expected error %v, got %v", tt.wantErr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// 2. Проверяем распарсенный Title
			if got.Title != tt.wantTitle {
				t.Errorf("\n[FAIL Title] %s\nПолучили: %q\nОжидали:  %q", tt.name, got.Title, tt.wantTitle)
			}

			// 3. Проверяем отделенное Body
			if got.Body != tt.wantBody {
				t.Errorf("\n[FAIL Body] %s\nПолучили:\n%q\nОжидали:\n%q", tt.name, got.Body, tt.wantBody)
			}
		})
	}
}
