package knowledge

import (
	"fmt"
	"net/url"
	"strings"
)

// Postgres describes a database connection in parts.
//
// An alternative to writing Config.DSN by hand. It exists because assembling
// that string correctly is easy to get wrong: a password containing @ / # or %
// has to be percent-encoded, and a DSN that is merely *mostly* right fails with
// a parse error about URL escapes rather than anything resembling
// "your password has a slash in it".
type Postgres struct {
	Host     string // default 127.0.0.1
	Port     string // default 5432
	User     string
	Password string
	Database string
	SSLMode  string // default disable
}

// DSN renders the connection string, escaping every part correctly.
func (p Postgres) DSN() string {
	host := p.Host
	if host == "" {
		host = "127.0.0.1"
	}
	port := p.Port
	if port == "" {
		port = "5432"
	}
	sslMode := p.SSLMode
	if sslMode == "" {
		sslMode = "disable"
	}

	u := url.URL{
		Scheme:   "postgres",
		User:     url.UserPassword(p.User, p.Password),
		Host:     host + ":" + port,
		Path:     p.Database,
		RawQuery: "sslmode=" + url.QueryEscape(sslMode),
	}
	return u.String()
}

func (p Postgres) validate() error {
	var missing []string
	if strings.TrimSpace(p.User) == "" {
		missing = append(missing, "User")
	}
	if strings.TrimSpace(p.Database) == "" {
		missing = append(missing, "Database")
	}
	if len(missing) > 0 {
		return fmt.Errorf("knowledge: Config.Postgres is missing %s", strings.Join(missing, ", "))
	}
	return nil
}
