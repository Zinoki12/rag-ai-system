package httpapi

import (
	"crypto/sha256"
	"crypto/subtle"
	"net"
	"net/http"
)

// publicPaths are served without a token even when one is configured.
//
// Only /health, and only because a monitoring probe is the one caller that
// usually cannot be given a credential. It answers with the service status, the
// active embedding space and three counts — nothing from the notes themselves.
// Every path that can return stored text, or spend money on an upstream model,
// is behind the token.
var publicPaths = map[string]bool{"/health": true}

// withAuth rejects requests that do not carry the shared token.
//
// A shared secret in a header is the weakest scheme worth having, and it is
// chosen deliberately: this service is meant to sit behind something else, and
// a real identity system belongs in that something else, not here. What matters
// is that the door is not simply open.
func withAuth(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	// Compare digests rather than the raw strings: equal-length digests make
	// the comparison independent of how much of the token the caller guessed
	// right, so a timing measurement leaks nothing about the secret. Comparing
	// the strings directly with == would return early at the first differing
	// byte and let an attacker recover the token one character at a time.
	want := sha256.Sum256([]byte(token))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if publicPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}

		got := sha256.Sum256([]byte(bearer(r.Header.Get("Authorization"))))
		if subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="rag"`)
			writeError(w, r, http.StatusUnauthorized, "missing or invalid token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// bearer extracts the credential from an Authorization header value. It returns
// "" for anything that is not a bearer token, which then fails the comparison
// like any other wrong value.
func bearer(header string) string {
	const prefix = "Bearer "
	if len(header) <= len(prefix) || !equalFold(header[:len(prefix)], prefix) {
		return ""
	}
	return header[len(prefix):]
}

// equalFold compares two ASCII strings case-insensitively. The scheme name in
// an Authorization header is case-insensitive per RFC 7235, and strings.EqualFold
// would additionally do Unicode folding that has no business here.
func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range len(a) {
		ca, cb := a[i], b[i]
		if 'A' <= ca && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if 'A' <= cb && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// reachableFromNetwork reports whether addr accepts connections from outside
// this machine. An unspecified address (0.0.0.0, ::) does; a loopback address
// does not. Anything this cannot classify is treated as reachable, because the
// safe default here is the restrictive one.
func reachableFromNetwork(addr net.Addr) bool {
	tcp, ok := addr.(*net.TCPAddr)
	if !ok {
		return true
	}
	return !tcp.IP.IsLoopback()
}
