//! App version shown in the menu bar and window title.
//!
//! Versioning rule: every big change adds 0.1 (0.4 -> 0.5 ...), and every
//! ten big changes bump the major number (0.9 -> 1.0). Bump `version` in
//! Cargo.toml and update `UPDATED` together.

/// "major.minor" from Cargo.toml (the patch number is not shown).
pub fn version() -> &'static str {
    let v = env!("CARGO_PKG_VERSION");
    match v.rmatch_indices('.').next() {
        Some((i, _)) if v[..i].contains('.') => &v[..i],
        _ => v,
    }
}

/// Date of the last big change.
pub const UPDATED: &str = "2026-11-11";

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_major_minor() {
        assert_eq!(super::version().matches('.').count(), 1);
    }
}
