// Package rag turns retrieved passages into a prompt.
//
// It is kept free of database and provider types on purpose: prompt assembly is
// the part of a RAG system most worth testing, and it should be testable
// without a Postgres container or an API key.
package rag

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Passage is one retrieved chunk, ready to be quoted to the model.
type Passage struct {
	Source string // note path, e.g. "rag/chunking.md"
	Title  string // note title from the frontmatter, may be empty
	Index  int    // which chunk of that note
	Text   string
}

// Ref identifies the exact chunk, for a citation the reader can go and check.
func (p Passage) Ref() string {
	return fmt.Sprintf("%s #%d", p.Source, p.Index)
}

// SystemPrompt constrains the model to the retrieved context.
//
// The instruction to admit missing information is the load-bearing line. A
// model asked to answer from context will otherwise fill a gap from its own
// training data, and the result is indistinguishable from a real answer — which
// defeats the point of grounding it in a private knowledge base at all.
const SystemPrompt = `Ты — ассистент по базе знаний. Отвечай на вопрос пользователя, опираясь ТОЛЬКО на приведённые ниже фрагменты.

Правила:
1. Если во фрагментах нет ответа — прямо скажи «В базе знаний нет ответа на этот вопрос» и не додумывай.
2. Не используй знания вне фрагментов, даже если уверен в них.
3. Ссылайся на фрагменты по их номеру в квадратных скобках, например [1].
4. Отвечай на языке вопроса, кратко и по делу.`

// DefaultMaxContextRunes bounds how much retrieved text goes into one prompt.
//
// Without a bound, a large top-k silently exceeds the model's context window:
// hosted APIs reject the call, and local models quietly truncate from the
// front, dropping the system instruction first.
const DefaultMaxContextRunes = 12000

// BuildUserPrompt renders the question and the numbered passages.
//
// Passages are added in the order given — best match first — until the budget
// runs out, so truncation drops the least relevant material rather than
// whatever happened to be last.
func BuildUserPrompt(question string, passages []Passage, maxContextRunes int) string {
	if maxContextRunes <= 0 {
		maxContextRunes = DefaultMaxContextRunes
	}

	var b strings.Builder
	b.WriteString("Фрагменты из базы знаний:\n\n")

	used := 0
	included := 0
	for i, p := range passages {
		text := strings.TrimSpace(p.Text)
		cost := utf8.RuneCountInString(text)
		if included > 0 && used+cost > maxContextRunes {
			break
		}

		header := p.Ref()
		if p.Title != "" {
			header = fmt.Sprintf("%s — %s", p.Title, p.Ref())
		}
		fmt.Fprintf(&b, "[%d] %s\n%s\n\n", i+1, header, text)

		used += cost
		included++
	}

	if included == 0 {
		b.WriteString("(ничего не найдено)\n\n")
	}

	fmt.Fprintf(&b, "Вопрос: %s", strings.TrimSpace(question))
	return b.String()
}

// Sources lists each passage's file once, in first-seen order.
//
// Deduplication is by note, not by chunk: several chunks of one file are one
// source to a reader, and listing the same path three times reads as noise.
func Sources(passages []Passage) []string {
	seen := make(map[string]struct{}, len(passages))
	var out []string
	for _, p := range passages {
		if _, dup := seen[p.Source]; dup {
			continue
		}
		seen[p.Source] = struct{}{}
		out = append(out, p.Source)
	}
	return out
}
