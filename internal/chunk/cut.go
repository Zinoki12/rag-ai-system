package chunk

import (
	"bufio"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

var ErrInvalidChunkSize = errors.New("chunk size must be greater than 0")

func Cut(text string, chunkSize int) ([]string, error) {
	if chunkSize <= 0 {
		return nil, ErrInvalidChunkSize
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return nil, nil
	}

	paragraphs, err := extractParagraphs(text)
	if err != nil {
		return nil, fmt.Errorf("extract paragraphs: %w", err)
	}

	if len(paragraphs) == 0 {
		return nil, nil
	}

	var chunks []string
	var currentChunk strings.Builder
	currentRunes := 0

	for _, p := range paragraphs {
		pRunes := utf8.RuneCountInString(p)

		if pRunes > chunkSize {
			if currentRunes > 0 {
				chunks = append(chunks, currentChunk.String())
				currentChunk.Reset()
				currentRunes = 0
			}

			parts := splitByRunes(p, chunkSize)
			chunks = append(chunks, parts...)
			continue
		}

		neededRunes := pRunes
		if currentRunes > 0 {
			neededRunes += 2
		}

		if currentRunes+neededRunes <= chunkSize {
			if currentRunes > 0 {
				currentChunk.WriteString("\n\n")
			}
			currentChunk.WriteString(p)
			currentRunes += neededRunes
		} else {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()

			currentChunk.WriteString(p)
			currentRunes = pRunes
		}
	}

	if currentRunes > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks, nil
}

func extractParagraphs(text string) ([]string, error) {
	scanner := bufio.NewScanner(strings.NewReader(text))
	var paragraphs []string
	var curPara strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			if curPara.Len() > 0 {
				paragraphs = append(paragraphs, strings.TrimSpace(curPara.String()))
				curPara.Reset()
			}
		} else {
			if curPara.Len() > 0 {
				curPara.WriteString("\n")
			}
			curPara.WriteString(line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner read line: %w", err)
	}

	if curPara.Len() > 0 {
		paragraphs = append(paragraphs, strings.TrimSpace(curPara.String()))
	}

	return paragraphs, nil
}

func splitByRunes(s string, limit int) []string {
	runes := []rune(s)
	var parts []string

	for len(runes) > limit {
		parts = append(parts, string(runes[:limit]))
		runes = runes[limit:]
	}
	if len(runes) > 0 {
		parts = append(parts, string(runes))
	}

	return parts
}
