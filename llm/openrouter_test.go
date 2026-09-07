package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenRouterNeedsAKey(t *testing.T) {
	_, err := NewOpenRouter("", "some/model", "", "")
	if err == nil {
		t.Fatal("провайдер собрался без ключа")
	}
	// The message has to name the variable and where a key comes from, because
	// this is the first thing a new user hits.
	for _, want := range []string{"OPENROUTER_API_KEY", "openrouter.ai/keys"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("в сообщении нет %q: %s", want, err)
		}
	}
}

func TestOpenRouterNeedsAModelAndSaysWhereToFindOne(t *testing.T) {
	_, err := NewOpenRouter("sk-or-v1-test", "", "", "")
	if err == nil {
		t.Fatal("провайдер собрался без модели")
	}
	for _, want := range []string{"LLM_MODEL", "openrouter.ai/models"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("в сообщении нет %q: %s", want, err)
		}
	}
}

// The endpoint is compiled in, so this is the only way to check it without
// making a network call: OpenRouter is the one provider a user cannot point
// somewhere else by mistake, and that is worth pinning down.
func TestOpenRouterTargetsTheBrokerAndAttributes(t *testing.T) {
	p, err := NewOpenRouter("sk-or-v1-test", "z-ai/glm-4.6:free", "https://example.org", "Журнал")
	if err != nil {
		t.Fatal(err)
	}
	o, ok := p.(*openAIProvider)
	if !ok {
		t.Fatalf("ожидался *openAIProvider, получен %T", p)
	}
	if o.baseURL != OpenRouterBaseURL {
		t.Errorf("baseURL = %q, ожидался %q", o.baseURL, OpenRouterBaseURL)
	}
	if got := o.extra["HTTP-Referer"]; got != "https://example.org" {
		t.Errorf("HTTP-Referer = %q", got)
	}
	if got := o.extra["X-Title"]; got != "Журнал" {
		t.Errorf("X-Title = %q", got)
	}
}

// Empty attribution must not turn into empty headers: OpenRouter treats a blank
// X-Title as a name, and the dashboard then shows a row with no label.
func TestOpenRouterOmitsBlankAttribution(t *testing.T) {
	p, err := NewOpenRouter("sk-or-v1-test", "z-ai/glm-4.6:free", "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	o := p.(*openAIProvider)
	if len(o.extra) != 0 {
		t.Errorf("ожидались пустые заголовки, получено %v", o.extra)
	}
}

// The headers must survive all the way onto the wire, not just into the struct.
func TestExtraHeadersReachTheServer(t *testing.T) {
	var gotReferer, gotTitle, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotReferer = r.Header.Get("HTTP-Referer")
		gotTitle = r.Header.Get("X-Title")
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{
				"role": "assistant", "content": "готово",
			}}},
		})
	}))
	defer srv.Close()

	p, err := newOpenAIProvider(srv.URL+"/v1", "sk-or-v1-test", "z-ai/glm-4.6:free",
		map[string]string{"HTTP-Referer": "https://example.org", "X-Title": "Журнал"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Generate(context.Background(), "", "вопрос"); err != nil {
		t.Fatal(err)
	}
	if gotReferer != "https://example.org" {
		t.Errorf("HTTP-Referer на сервере = %q", gotReferer)
	}
	if gotTitle != "Журнал" {
		t.Errorf("X-Title на сервере = %q", gotTitle)
	}
	if gotAuth != "Bearer sk-or-v1-test" {
		t.Errorf("Authorization на сервере = %q", gotAuth)
	}
}

func TestConfigFromEnvPrefersTheOpenRouterKey(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "openrouter")
	t.Setenv("LLM_MODEL", "z-ai/glm-4.6:free")
	t.Setenv("OPENROUTER_API_KEY", "sk-or-v1-specific")
	t.Setenv("LLM_API_KEY", "sk-generic")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "sk-or-v1-specific" {
		t.Errorf("взят ключ %q, ожидался OPENROUTER_API_KEY", cfg.APIKey)
	}
	if cfg.AppURL == "" || cfg.AppName == "" {
		t.Errorf("атрибуция не заполнена по умолчанию: %q / %q", cfg.AppURL, cfg.AppName)
	}
}

// Falling back keeps the single-provider case from needing two variables that
// hold the same string.
func TestConfigFromEnvFallsBackToTheGenericKey(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "openrouter")
	t.Setenv("LLM_MODEL", "z-ai/glm-4.6:free")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("LLM_API_KEY", "sk-generic")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "sk-generic" {
		t.Errorf("взят ключ %q, ожидался LLM_API_KEY", cfg.APIKey)
	}
}

// A Google key must never travel to OpenRouter's host.
func TestConfigFromEnvDoesNotSendTheGoogleKeyToOpenRouter(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "openrouter")
	t.Setenv("LLM_MODEL", "z-ai/glm-4.6:free")
	t.Setenv("GOOGLE_API_KEY", "AIza-secret")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("LLM_API_KEY", "")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(cfg.APIKey, "AIza") {
		t.Fatalf("ключ Google утёк в конфигурацию OpenRouter: %q", cfg.APIKey)
	}
}
