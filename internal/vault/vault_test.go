package vault

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// writeVault materialises a vault in a temporary directory.
//
// Previously this test read the repository's own vault/ and asserted a
// hardcoded count of 5, so adding a note to the vault broke the test suite and
// the test could not describe the cases it was actually covering. A fixture
// built in the test states its own preconditions.
func writeVault(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()

	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir for %s: %v", name, err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return root
}

func TestScan(t *testing.T) {
	root := writeVault(t, map[string]string{
		"top.md":               "---\ntitle: Верхний\n---\nТело верхней заметки.",
		"nested/deep/inner.md": "---\ntitle: Вложенный\n---\nТело вложенной заметки.",
		"кириллица в имени.md": "Без фронтматтера вообще.",
		"upper.MD":             "---\ntitle: Регистр\n---\nРасширение заглавными.",
		"notes.txt":            "не markdown, не должен попасть",
		"assets/picture.png":   "\x89PNG\r\n",
		"nested/README":        "без расширения",
	})

	scan, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() unexpected error: %v", err)
	}
	files := scan.Notes

	if want := 4; len(files) != want {
		paths := make([]string, len(files))
		for i, f := range files {
			paths[i] = f.Path
		}
		t.Fatalf("Scan() returned %d files %v, want %d", len(files), paths, want)
	}

	byPath := make(map[string]bool, len(files))
	for _, f := range files {
		byPath[filepath.ToSlash(f.Path)] = true

		if f.Path == "" {
			t.Error("file path is empty")
		}
		if filepath.IsAbs(f.Path) {
			t.Errorf("path %q is absolute, want relative to the vault root", f.Path)
		}
		if f.Hash == [32]byte{} {
			t.Errorf("hash is empty for %s", f.Path)
		}
		if f.Text == "" {
			t.Errorf("text is empty for %s", f.Path)
		}
	}

	for _, want := range []string{"top.md", "nested/deep/inner.md", "кириллица в имени.md", "upper.MD"} {
		if !byPath[want] {
			t.Errorf("%s was not scanned", want)
		}
	}
	for _, unwanted := range []string{"notes.txt", "assets/picture.png", "nested/README"} {
		if byPath[unwanted] {
			t.Errorf("%s was scanned but is not markdown", unwanted)
		}
	}
}

func TestScanTitleAndBody(t *testing.T) {
	root := writeVault(t, map[string]string{
		"a.md": "---\ntitle: Заголовок\n---\nПервый абзац.\n\nВторой абзац.",
	})

	scan, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan(): %v", err)
	}
	files := scan.Notes
	if len(files) != 1 {
		t.Fatalf("got %d files, want 1", len(files))
	}

	if got := files[0].Name; got != "Заголовок" {
		t.Errorf("Name = %q, want %q", got, "Заголовок")
	}
	if got := files[0].Text; got != "Первый абзац.\n\nВторой абзац." {
		t.Errorf("Text = %q, frontmatter was not stripped cleanly", got)
	}
}

// A file with an unterminated frontmatter block is a warning, not a failure:
// one broken note must not cost the whole vault.
func TestScanReportsBadFrontmatterAsWarning(t *testing.T) {
	root := writeVault(t, map[string]string{
		"good.md":   "---\ntitle: Хорошая\n---\nТело.",
		"broken.md": "---\ntitle: Незакрытая\nникакого закрывающего разделителя",
	})

	scan, err := Scan(root)
	if err == nil {
		t.Fatal("Scan() reported no warning for unterminated frontmatter")
	}
	if !errors.Is(err, ErrUnclosedFrontmatter) {
		t.Fatalf("Scan() error = %v, want it to wrap ErrUnclosedFrontmatter", err)
	}
	if len(scan.Notes) != 1 {
		t.Errorf("got %d notes, want the one good note to survive", len(scan.Notes))
	}

	// The skipped file must be named, not merely counted. A caller that prunes
	// the index against the scan needs to know this path is still on disk;
	// without it the broken note is indistinguishable from a deleted one and
	// gets erased from the index along with its vectors.
	want := []string{"broken.md"}
	if !slices.Equal(scan.Skipped, want) {
		t.Errorf("Skipped = %v, want %v", scan.Skipped, want)
	}
}

func TestScanEmptyVault(t *testing.T) {
	scan, err := Scan(t.TempDir())
	if err != nil {
		t.Fatalf("Scan() on an empty directory: %v", err)
	}
	if len(scan.Notes) != 0 {
		t.Errorf("got %d notes, want 0", len(scan.Notes))
	}
}

func TestScanMissingRoot(t *testing.T) {
	if _, err := Scan(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Error("Scan() on a missing directory returned no error")
	}
}
