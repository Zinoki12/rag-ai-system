package vault

import (
	"path/filepath"
	"testing"
)

func TestScan(t *testing.T) {
	vaultRoot := filepath.Join("..", "..", "vault")

	files, err := Scan(vaultRoot)
	if err != nil {
		// В чистом vault не должно быть даже предупреждений
		t.Fatalf("Scan() unexpected error: %v", err)
	}

	expectedCount := 5
	if len(files) != expectedCount {
		t.Fatalf("expected %d .md files, got %d", expectedCount, len(files))
	}

	for _, f := range files {
		if f.Path == "" {
			t.Errorf("file path is empty")
		}
		if f.Hash == [32]byte{} {
			t.Errorf("hash is empty for file %s", f.Path)
		}
		if f.Text == "" {
			t.Errorf("text is empty for file %s", f.Path)
		}
		if filepath.Ext(f.Path) != ".md" {
			t.Errorf("non-md file scanned: %s", f.Path)
		}
	}
}
