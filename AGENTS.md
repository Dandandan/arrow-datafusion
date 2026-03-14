# Agent Guidelines for Apache DataFusion

## Developer Documentation

- [Contributor Guide](https://datafusion.apache.org/contributor-guide/index.html)
- [Architecture Guide](https://datafusion.apache.org/contributor-guide/architecture.html)

## Before Committing

Before committing any changes, you **must** run the following checks and fix any issues:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

- `cargo fmt` ensures consistent code formatting across the project.
- `cargo clippy` catches common mistakes and enforces idiomatic Rust patterns. All warnings must be resolved (treated as errors via `-D warnings`).

Do not commit code that fails either of these checks.

## Testing

Run relevant tests before submitting changes:

```bash
cargo test --all-features
```

For SQL logic tests:

```bash
cargo test -p datafusion-sqllogictest
```
