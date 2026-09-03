// Package config reads settings from the environment.
//
// Every lookup goes through here so that a typo in a numeric variable fails at
// startup with the variable's name in the message, rather than silently
// falling back to a default and being discovered later as "why is it slow".
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// String returns the variable's value, or def when it is unset or blank.
func String(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// Int returns the variable parsed as an int, or def when it is unset.
func Int(key string, def int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def, nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s=%q: not an integer", key, raw)
	}
	return v, nil
}

// Float returns the variable parsed as a float64, or def when it is unset.
func Float(key string, def float64) (float64, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("%s=%q: not a number", key, raw)
	}
	return v, nil
}

// Required returns the variable's value or an error naming it.
func Required(key string) (string, error) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return "", fmt.Errorf("%s is not set", key)
	}
	return v, nil
}

// Bool returns the variable parsed as a bool, or def when it is unset.
func Bool(key string, def bool) (bool, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def, nil
	}
	v, err := strconv.ParseBool(raw)
	if err != nil {
		return false, fmt.Errorf("%s=%q: not a boolean (try true or false)", key, raw)
	}
	return v, nil
}
