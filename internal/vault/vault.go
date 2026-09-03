package vault

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Zinoki12/rag-ai-system/internal/model"
)

// Result is what one pass over the vault found.
//
// Skipped is not a detail for logs. A file that is on disk but could not be
// parsed is still a file that exists, and a caller pruning the index against
// Notes alone would delete it — losing an indexed note because someone opened
// it in an editor and broke its frontmatter. Callers that prune must treat
// Skipped as present.
type Result struct {
	Notes   []*model.Note
	Skipped []string // vault-relative paths, in the same form as Note.Path
}

// Scan walks root and parses every markdown file under it.
//
// The returned error carries two different things and callers must tell them
// apart. A file this scan had to skip is reported as a warning joined into the
// error and its path is listed in Result.Skipped, while the rest of the vault
// is returned normally. A failure of the walk itself — an unreadable directory,
// a vanished file — comes back with an empty Result and means nothing about the
// vault can be trusted. errors.Is(err, ErrUnclosedFrontmatter) distinguishes
// the first kind.
func Scan(root string) (Result, error) {
	var res Result
	var warnErrs []error

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			wrappedErr := fmt.Errorf("error at WalkDir at %s: %w", path, err)
			return wrappedErr
		}

		if !d.IsDir() && strings.ToLower(filepath.Ext(path)) == ".md" {
			relPath, err := filepath.Rel(root, path)
			if err != nil {
				wrappedErr := fmt.Errorf("error at rel filepath %s: %w", path, err)
				return wrappedErr
			}

			content, err := os.ReadFile(path)
			if err != nil {
				wrappedErr := fmt.Errorf("error at readfile %s: %w", path, err)
				return wrappedErr
			}
			text := string(content)

			yaml, err := yamlParser(text)
			if err != nil {
				if errors.Is(err, ErrUnclosedFrontmatter) {
					warnErrs = append(warnErrs, fmt.Errorf("skip %s: %w", path, err))
					res.Skipped = append(res.Skipped, relPath)
					return nil
				}
				return fmt.Errorf("parse yaml in %s: %w", path, err)
			}

			fileInfo := &model.Note{
				Path: relPath,
				Name: yaml.Title,
				Text: yaml.Body,
				Hash: sha256.Sum256(content),
			}
			res.Notes = append(res.Notes, fileInfo)
		}

		return nil
	})
	if err != nil {
		return Result{}, fmt.Errorf("error walking the path %q: %w", root, err)
	}
	return res, errors.Join(warnErrs...)
}
