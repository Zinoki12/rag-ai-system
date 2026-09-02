package vault

import (
	"bufio"
	"errors"
	"fmt"
	"strings"
)

type yamlInfo struct {
	Title string
	Body  string
}

const (
	maxBuffSize = 10 * 1024 * 1024
)

var ErrUnclosedFrontmatter = errors.New("unclosed frontmatter")

func yamlParser(text string) (yamlInfo, error) {
	// Убираем BOM, если он есть
	text = strings.TrimPrefix(text, "\ufeff")

	scanner := bufio.NewScanner(strings.NewReader(text))
	initBuff := make([]byte, 256*1024)
	scanner.Buffer(initBuff, maxBuffSize)
	yaml := yamlInfo{}
	var body strings.Builder

	foundStart := false
	inFrontmatter := false
	bodyStarted := false

	for scanner.Scan() {
		rawLine := scanner.Text()
		trimmedLine := strings.TrimSpace(rawLine)

		if bodyStarted {
			body.WriteString(rawLine)
			body.WriteString("\n")
			continue
		}

		if !foundStart {
			if trimmedLine == "" {
				continue
			}
			if trimmedLine == "---" {
				foundStart = true
				inFrontmatter = true
				continue
			}

			yaml.Body = strings.TrimSpace(text)
			return yaml, nil
		}

		if inFrontmatter && trimmedLine == "---" {
			inFrontmatter = false
			bodyStarted = true
			continue
		}

		if inFrontmatter {
			key, val, ok := strings.Cut(trimmedLine, ":")
			if !ok {
				continue
			}

			key = strings.ToLower(strings.TrimSpace(key))
			val = strings.TrimSpace(val)
			val = strings.Trim(val, `"'`)

			switch key {
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

	yaml.Body = strings.TrimSpace(body.String())
	return yaml, nil
}
