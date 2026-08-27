package model

type Note struct {
	Path string
	Hash [32]byte
	Text string
	Name string
}
