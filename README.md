# ⚖️☕ LegalKofi

## 📑 A RAG with Cross-Encoders Re-ranking Application

An LLM app with RAG for the legal professionals in Ghana.

🚨 NOTE: **Requires `Python > 3.10` with  `SQLite > 3.35`**

## 🤖 Prerequisites

- [Ollama](https://ollama.dev/download)

## 🔨 Setting up locally

Create virtualenv and install dependencies.

```sh
make setup
```

## ⚡️ Running the application

```sh
make run
```

## ✨ Linters and Formatters

Check for linting rule violations:

```sh
make check
```

Auto-fix linting violations:

```sh
make fix
```

## 🤸‍♀️ Getting Help

```sh
make

# OR

make help
```

## 🔧 Common Issues and Fixes

- If you run into any errors with incompatible version of ChromaDB/Sqlite3, refer to [this solution](https://docs.trychroma.com/troubleshooting#sqlite).
