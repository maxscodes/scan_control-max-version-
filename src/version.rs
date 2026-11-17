//! App version shown in the menu bar and window title.
//!
//! Versioning rule: each day with changes adds 0.1 (0.9 -> 1.0 -> 1.1 ...),
//! however many changes that day; more changes on the same day keep the
//! version. On the first change of a new day, bump `version` in Cargo.toml
//! and set `UPDATED` to that day.

/// "major.minor" from Cargo.toml (the patch number is not shown).
pub fn version() -> &'static str {
    let v = env!("CARGO_PKG_VERSION");
    match v.rmatch_indices('.').next() {
        Some((i, _)) if v[..i].contains('.') => &v[..i],
        _ => v,
    }
}

/// Last day with changes.
pub const UPDATED: &str = "2026-11-17";

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_major_minor() {
        assert_eq!(super::version().matches('.').count(), 1);
    }
}
