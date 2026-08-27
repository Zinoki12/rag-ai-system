package vault

import (
	"bufio"
	"errors"
	"fmt"
	"strings"
)

type yamlInfo struct {
	Title string
}

var ErrUnclosedFrontmatter = errors.New("unclosed frontmatter")

func yamlParser(text string) (yamlInfo, error) {
	scanner := bufio.NewScanner(strings.NewReader(text))
	yaml := yamlInfo{}
	lineNum := 0
	inFrontmatter := false

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		if lineNum == 1 {
			if line == "---" {
				inFrontmatter = true
				continue
			}
			return yaml, nil
		}

		if inFrontmatter && line == "---" {
			inFrontmatter = false
			break
		}

		if inFrontmatter {
			key, val, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}

			key = strings.TrimSpace(key)
			val = strings.TrimSpace(val)
			val = strings.Trim(val, `"'`)

			switch strings.ToLower(key) {
			case "title":
				yaml.Title = val
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return yamlInfo{}, fmt.Errorf("scanner error: %w", err)
	}

	if inFrontmatter {
		return yamlInfo{}, ErrUnclosedFrontmatter
	}

	return yaml, nil
}
