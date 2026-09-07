package llm

import (
	"fmt"
	"strings"
)

// ProviderOpenRouter is the value of LLM_PROVIDER that selects OpenRouter.
//
// OpenRouter is a broker: one key and one endpoint in front of models from
// many vendors, including a rotating set offered free of charge. Its API is
// OpenAI-compatible, so this is not a second client — it is NewOpenAI with the
// endpoint filled in and two headers added.
const ProviderOpenRouter = "openrouter"

// OpenRouterBaseURL is the endpoint every OpenRouter call goes to.
const OpenRouterBaseURL = "https://openrouter.ai/api/v1"

// NewOpenRouter builds a generation provider backed by OpenRouter.
//
// referer and title are optional. OpenRouter reads them to attribute traffic to
// an application; sending them is what puts a name next to the usage instead of
// "unknown", and neither affects the answer. They are arguments rather than
// constants because this is a library: the application that links it knows its
// own name, and this package does not.
func NewOpenRouter(apiKey, model, referer, title string) (Provider, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, fmt.Errorf(
			"openrouter generation: no API key (set OPENROUTER_API_KEY; " +
				"a key is issued at https://openrouter.ai/keys)")
	}
	if strings.TrimSpace(model) == "" {
		// No default model on purpose. Which models OpenRouter offers, and
		// which of them cost nothing, changes month to month; a name compiled
		// in here would eventually 404 and read like a bug in this code rather
		// than a retired model. Naming the catalogue is more useful than
		// guessing an entry in it.
		return nil, fmt.Errorf(
			"openrouter generation: no model (set LLM_MODEL, " +
				"for example LLM_MODEL=\"z-ai/glm-4.6:free\"; " +
				"the free ones are listed at https://openrouter.ai/models?max_price=0)")
	}

	extra := map[string]string{}
	if strings.TrimSpace(referer) != "" {
		extra["HTTP-Referer"] = referer
	}
	if strings.TrimSpace(title) != "" {
		extra["X-Title"] = title
	}
	return newOpenAIProvider(OpenRouterBaseURL, apiKey, model, extra)
}
