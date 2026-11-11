//! Look and feel: light and dark palettes, status colours, the shared window
//! frame, small widgets (status pill), and persisted UI preferences.

use std::sync::{LazyLock, Mutex};

use egui::{
    epaint::Shadow, Color32, CornerRadius, FontFamily, FontId, Frame, Margin, Stroke, Style,
    TextStyle, Theme, ThemePreference, Ui, Vec2,
};
use serde::{Deserialize, Serialize};

/// Status colours, readable on both themes.
pub const OK: Color32 = Color32::from_rgb(0x2e, 0xb8, 0x5c);
pub const WARN: Color32 = Color32::from_rgb(0xe8, 0xa3, 0x17);
pub const ERR: Color32 = Color32::from_rgb(0xd9, 0x48, 0x48);
pub const IDLE: Color32 = Color32::from_rgb(0x8a, 0x93, 0x9e);

const ACCENT_DARK: Color32 = Color32::from_rgb(0x3d, 0xa5, 0xd9);
const ACCENT_LIGHT: Color32 = Color32::from_rgb(0x1f, 0x6f, 0xb5);

/// Accent colour for the current theme.
pub fn accent(ui: &Ui) -> Color32 {
    if ui.visuals().dark_mode {
        ACCENT_DARK
    } else {
        ACCENT_LIGHT
    }
}

/// Configure both the light and dark styles once at start-up, and apply the
/// saved theme choice. Switching later only needs [`set_theme`].
pub fn install(ctx: &egui::Context) {
    ctx.style_mut_of(Theme::Dark, |s| style(s, true));
    ctx.style_mut_of(Theme::Light, |s| style(s, false));
    ctx.set_theme(ThemePreference::from(prefs().theme));
}

/// Switch theme and remember the choice.
pub fn set_theme(ctx: &egui::Context, theme: ThemeChoice) {
    ctx.set_theme(ThemePreference::from(theme));
    update_prefs(|p| p.theme = theme);
}

/// Toggle between light and dark (resolving System to what is shown now).
pub fn toggle_theme(ctx: &egui::Context) {
    let next = match ctx.theme() {
        Theme::Dark => ThemeChoice::Light,
        Theme::Light => ThemeChoice::Dark,
    };
    set_theme(ctx, next);
}

fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

fn style(s: &mut Style, dark: bool) {
    let accent = if dark { ACCENT_DARK } else { ACCENT_LIGHT };
    let v = &mut s.visuals;
    if dark {
        v.panel_fill = rgb(0x1b, 0x1e, 0x23);
        v.window_fill = rgb(0x22, 0x26, 0x2c);
        v.extreme_bg_color = rgb(0x14, 0x16, 0x1a);
        v.faint_bg_color = rgb(0x27, 0x2b, 0x32);
        v.code_bg_color = rgb(0x2a, 0x2f, 0x37);
        v.window_stroke = Stroke::new(1., rgb(0x36, 0x3c, 0x45));
        let w = &mut v.widgets;
        w.noninteractive.bg_fill = v.window_fill;
        w.noninteractive.weak_bg_fill = v.window_fill;
        w.noninteractive.bg_stroke = Stroke::new(1., rgb(0x33, 0x39, 0x41));
        w.noninteractive.fg_stroke = Stroke::new(1., rgb(0xc9, 0xd1, 0xd9));
        w.inactive.bg_fill = rgb(0x2f, 0x34, 0x3c);
        w.inactive.weak_bg_fill = rgb(0x2f, 0x34, 0x3c);
        w.inactive.fg_stroke = Stroke::new(1., rgb(0xd6, 0xdd, 0xe4));
        w.hovered.bg_fill = rgb(0x3a, 0x41, 0x4b);
        w.hovered.weak_bg_fill = rgb(0x3a, 0x41, 0x4b);
        w.hovered.bg_stroke = Stroke::new(1., accent.gamma_multiply(0.7));
        w.active.bg_fill = rgb(0x2a, 0x5f, 0x80);
        w.active.weak_bg_fill = rgb(0x2a, 0x5f, 0x80);
        w.open.weak_bg_fill = rgb(0x33, 0x39, 0x41);
    } else {
        v.panel_fill = rgb(0xf3, 0xf4, 0xf6);
        v.window_fill = rgb(0xfb, 0xfb, 0xfc);
        v.extreme_bg_color = rgb(0xff, 0xff, 0xff);
        v.faint_bg_color = rgb(0xee, 0xf0, 0xf3);
        v.code_bg_color = rgb(0xe9, 0xec, 0xf0);
        v.window_stroke = Stroke::new(1., rgb(0xd3, 0xd7, 0xdd));
        let w = &mut v.widgets;
        w.noninteractive.bg_fill = v.window_fill;
        w.noninteractive.weak_bg_fill = v.window_fill;
        w.noninteractive.bg_stroke = Stroke::new(1., rgb(0xdd, 0xe1, 0xe6));
        w.noninteractive.fg_stroke = Stroke::new(1., rgb(0x2b, 0x31, 0x38));
        w.inactive.bg_fill = rgb(0xe6, 0xe9, 0xed);
        w.inactive.weak_bg_fill = rgb(0xe6, 0xe9, 0xed);
        w.inactive.fg_stroke = Stroke::new(1., rgb(0x1f, 0x25, 0x2b));
        w.hovered.bg_fill = rgb(0xdb, 0xe4, 0xee);
        w.hovered.weak_bg_fill = rgb(0xdb, 0xe4, 0xee);
        w.hovered.bg_stroke = Stroke::new(1., accent.gamma_multiply(0.7));
        w.active.bg_fill = rgb(0xc4, 0xd9, 0xee);
        w.active.weak_bg_fill = rgb(0xc4, 0xd9, 0xee);
        w.open.weak_bg_fill = rgb(0xe1, 0xe5, 0xea);
    }
    v.selection.bg_fill = accent.gamma_multiply(if dark { 0.55 } else { 0.35 });
    v.selection.stroke = Stroke::new(1., accent);
    v.hyperlink_color = accent;
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(5);
    }
    let shadow = Color32::from_black_alpha(if dark { 90 } else { 40 });
    v.window_shadow = Shadow { offset: [0, 4], blur: 16, spread: 0, color: shadow };
    v.popup_shadow = Shadow { offset: [0, 3], blur: 10, spread: 0, color: shadow };

    s.spacing.item_spacing = Vec2::new(8., 6.);
    s.spacing.button_padding = Vec2::new(8., 4.);
    s.spacing.interact_size.y = 22.;
    s.spacing.window_margin = Margin::same(10);
    s.text_styles = [
        (TextStyle::Heading, FontId::new(18., FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(11., FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
    ]
    .into();
}

/// Frame for tool windows (Sequence, Spectroscopy, Lithography, ...).
pub fn window_frame(ctx: &egui::Context) -> Frame {
    Frame::window(&ctx.style())
}

/// Translucent frame for panels that float over the map (Scan Control, Layers).
pub fn overlay_frame(ctx: &egui::Context) -> Frame {
    Frame::window(&ctx.style()).multiply_with_opacity(0.92)
}

/// A coloured status dot with a label. `pulse` animates the dot (scanning).
pub fn status_pill(ui: &mut Ui, color: Color32, text: &str, pulse: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(14.), egui::Sense::hover());
    let mut c = color;
    if pulse {
        let t = ui.input(|i| i.time);
        let a = 0.55 + 0.45 * (t * 3.0).sin().abs() as f32;
        c = color.gamma_multiply(a);
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
    }
    ui.painter().circle_filled(rect.center(), 5., c);
    ui.label(egui::RichText::new(text).strong().color(color));
}

/// Saved theme choice (egui's own type isn't serialisable in this build).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeChoice {
    Dark,
    Light,
    System,
}
impl From<ThemeChoice> for ThemePreference {
    fn from(t: ThemeChoice) -> Self {
        match t {
            ThemeChoice::Dark => ThemePreference::Dark,
            ThemeChoice::Light => ThemePreference::Light,
            ThemeChoice::System => ThemePreference::System,
        }
    }
}

/// UI preferences kept between runs.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiPrefs {
    pub theme: ThemeChoice,
    pub scan_compact: bool,
    pub sequence_details: bool,
    pub show_grid: bool,
}
impl Default for UiPrefs {
    fn default() -> Self {
        Self {
            theme: ThemeChoice::Dark,
            scan_compact: false,
            sequence_details: true,
            show_grid: false,
        }
    }
}

fn prefs_path() -> std::path::PathBuf {
    crate::app::CONFIG_DIR.config_local_dir().join("ui_prefs.json")
}

static PREFS: LazyLock<Mutex<UiPrefs>> = LazyLock::new(|| {
    Mutex::new(
        std::fs::read_to_string(prefs_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
    )
});

/// Current preferences (a copy).
pub fn prefs() -> UiPrefs {
    PREFS.lock().map(|p| p.clone()).unwrap_or_default()
}

/// Change preferences and save them if anything changed.
pub fn update_prefs(f: impl FnOnce(&mut UiPrefs)) {
    let Ok(mut p) = PREFS.lock() else { return };
    let before = serde_json::to_string(&*p).ok();
    f(&mut p);
    let after = serde_json::to_string(&*p).ok();
    if before != after {
        if let Some(json) = after {
            if let Err(e) = std::fs::write(prefs_path(), json) {
                tracing::warn!("could not save UI preferences: {e}");
            }
        }
    }
}
