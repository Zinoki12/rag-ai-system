package embed

import (
	"context"
	"regexp"
	"strings"
	"testing"
)

var slugSafe = regexp.MustCompile(`^[a-z0-9_]+$`)

// These are golden values on purpose. The slug is a physical table name: if it
// ever changes for the same space, every vector already stored under the old
// name is orphaned and silently recomputed into a new table. Changing an
// expectation here should be a deliberate act with a data migration attached.
func TestSpaceSlug(t *testing.T) {
	tests := []struct {
		name  string
		space Space
		want  string
	}{
		{
			name:  "обычная модель Google",
			space: Space{Provider: "google", Model: "gemini-embedding-001", Dim: 768},
			want:  "emb_google__gemini_embedding_001__768_ab5787e4",
		},
		{
			name:  "тег модели Ollama через двоеточие",
			space: Space{Provider: "ollama", Model: "nomic-embed-text:latest", Dim: 768},
			want:  "emb_ollama__nomic_embed_text_latest__768_7bd74f9f",
		},
		{
			name:  "попытка SQL-инъекции в имени модели",
			space: Space{Provider: "ollama", Model: `x"; DROP TABLE notes; --`, Dim: 8},
			want:  "emb_ollama__x_drop_table_notes__8_66bbc9b4",
		},
		{
			name:  "кириллица вычищается, но пространство остаётся различимым",
			space: Space{Provider: "ollama", Model: "модель раз", Dim: 4},
			want:  "emb_ollama____4_d8a18031",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.space.Slug()
			if got != tt.want {
				t.Errorf("Slug() = %q, want %q", got, tt.want)
			}
			if !slugSafe.MatchString(got) {
				t.Errorf("Slug() = %q contains characters outside [a-z0-9_]", got)
			}
		})
	}
}

// Two spaces must never share a table, however the readable part of the name
// gets mangled. Sharing one would file two incompatible vector spaces into a
// single index and ruin every search that touched it — with no error anywhere.
func TestSpaceSlugStaysDistinct(t *testing.T) {
	longPrefix := strings.Repeat("verylongmodelname", 4)

	tests := []struct {
		name string
		a, b Space
	}{
		{
			name: "длинные имена с общим префиксом (обрезка)",
			a:    Space{Provider: "ollama", Model: longPrefix + "-alpha", Dim: 768},
			b:    Space{Provider: "ollama", Model: longPrefix + "-beta", Dim: 768},
		},
		{
			name: "имена целиком из кириллицы (санитизация в пустоту)",
			a:    Space{Provider: "ollama", Model: "модель раз", Dim: 4},
			b:    Space{Provider: "ollama", Model: "модель два", Dim: 4},
		},
		{
			name: "дефис против подчёркивания (свёртка разделителей)",
			a:    Space{Provider: "ollama", Model: "some-model", Dim: 768},
			b:    Space{Provider: "ollama", Model: "some_model", Dim: 768},
		},
		{
			name: "одна модель, разные размерности",
			a:    Space{Provider: "google", Model: "gemini-embedding-001", Dim: 768},
			b:    Space{Provider: "google", Model: "gemini-embedding-001", Dim: 1536},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.a.Slug() == tt.b.Slug() {
				t.Fatalf("distinct spaces collided on one slug: %q", tt.a.Slug())
			}
			for _, s := range []Space{tt.a, tt.b} {
				if got := len(s.Slug()); got > maxSlugLen {
					t.Errorf("Slug() length = %d, want <= %d (%q)", got, maxSlugLen, s.Slug())
				}
				if !slugSafe.MatchString(s.Slug()) {
					t.Errorf("Slug() = %q contains characters outside [a-z0-9_]", s.Slug())
				}
			}
		})
	}
}

func TestSpaceValidate(t *testing.T) {
	tests := []struct {
		name    string
		space   Space
		wantErr bool
	}{
		{"валидное пространство", Space{Provider: "ollama", Model: "m", Dim: 768}, false},
		{"пустой провайдер", Space{Model: "m", Dim: 768}, true},
		{"провайдер из пробелов", Space{Provider: "   ", Model: "m", Dim: 768}, true},
		{"пустая модель", Space{Provider: "ollama", Dim: 768}, true},
		{"нулевая размерность", Space{Provider: "ollama", Model: "m"}, true},
		{"отрицательная размерность", Space{Provider: "ollama", Model: "m", Dim: -1}, true},
		{"размерность за пределом pgvector", Space{Provider: "ollama", Model: "m", Dim: 20000}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.space.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckResult(t *testing.T) {
	space := Space{Provider: "fake", Model: "m", Dim: 3}

	tests := []struct {
		name    string
		texts   []string
		got     []Vector
		wantErr bool
	}{
		{"всё сходится", []string{"a", "b"}, []Vector{{1, 2, 3}, {4, 5, 6}}, false},
		{"провайдер потерял вектор", []string{"a", "b"}, []Vector{{1, 2, 3}}, true},
		{"провайдер вернул лишний вектор", []string{"a"}, []Vector{{1, 2, 3}, {4, 5, 6}}, true},
		{"чужая размерность", []string{"a"}, []Vector{{1, 2}}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkResult(space, tt.texts, tt.got)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkResult() error = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

// The fake provider is what the offline end-to-end tests run on, so its
// contract is worth pinning down.
func TestFakeProvider(t *testing.T) {
	const dim = 64
	p, err := NewFake(dim)
	if err != nil {
		t.Fatalf("NewFake: %v", err)
	}

	ctx := context.Background()

	t.Run("пустой вход даёт пустой выход без ошибки", func(t *testing.T) {
		got, err := p.Embed(ctx, nil, KindDocument)
		if err != nil || len(got) != 0 {
			t.Fatalf("Embed(nil) = %v, %v; want empty, nil", got, err)
		}
	})

	t.Run("размерность и порядок сохраняются", func(t *testing.T) {
		got, err := p.Embed(ctx, []string{"первый текст", "второй текст"}, KindDocument)
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("got %d vectors, want 2", len(got))
		}
		for i, v := range got {
			if len(v) != dim {
				t.Errorf("vector %d has dimension %d, want %d", i, len(v), dim)
			}
		}
		if cosine(got[0], got[1]) > 0.999 {
			t.Error("different texts produced effectively identical vectors")
		}
	})

	t.Run("детерминированность", func(t *testing.T) {
		a, _ := p.Embed(ctx, []string{"один и тот же текст"}, KindDocument)
		b, _ := p.Embed(ctx, []string{"один и тот же текст"}, KindDocument)
		if cosine(a[0], b[0]) < 0.9999 {
			t.Error("same input produced different vectors across calls")
		}
	})

	t.Run("похожие тексты ближе, чем непохожие", func(t *testing.T) {
		vs, err := p.Embed(ctx, []string{
			"чанкинг режет текст на куски",
			"чанкинг режет текст на части",
			"миграции накатываются через goose",
		}, KindDocument)
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if near, far := cosine(vs[0], vs[1]), cosine(vs[0], vs[2]); near <= far {
			t.Errorf("similar texts scored %.3f, unrelated scored %.3f; want similar to score higher", near, far)
		}
	})

	t.Run("отменённый контекст возвращает ошибку", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := p.Embed(cancelled, []string{"текст"}, KindDocument); err == nil {
			t.Error("Embed() with a cancelled context returned no error")
		}
	})
}

func cosine(a, b Vector) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt(na) * sqrt(nb))
}

func sqrt(f float64) float64 {
	if f <= 0 {
		return 0
	}
	x := f
	for range 40 {
		x = (x + f/x) / 2
	}
	return x
}
