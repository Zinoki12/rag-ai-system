package vault

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type FileInfo struct {
	Path string
	Name string
	Text string
	Hash [32]byte
	Date string
}

// This function returns a slice of file's info and error, but errors actually 2 types:
// 1. Simple warning about crashed file
// 2. Critical error (like crash system, harddrive , etc...)
func Scan(root string) ([]*FileInfo, error) {
	var files []*FileInfo
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
					return nil
				}
				return fmt.Errorf("parse yaml in %s: %w", path, err)
			}

			fileInfo := &FileInfo{
				Path: relPath,
				Name: yaml.Title,
				Text: text,
				Hash: sha256.Sum256(content),
				Date: yaml.Created,
			}
			files = append(files, fileInfo)
		}

		return nil
	})
	if err != nil {
		return []*FileInfo{}, fmt.Errorf("error walking the path %q: %w", root, err)
	}
	return files, errors.Join(warnErrs...)
}
