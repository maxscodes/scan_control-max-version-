use std::{
    hash::Hash,
    ops::{Deref, DerefMut},
};

use egui::{
    AtomExt as _, AtomKind, AtomLayout, AtomLayoutResponse, Atoms, Context, FontSelection, Frame,
    Layout, Response, Sense, TextStyle, Ui,
};
use itertools::Itertools;

use crate::utils::response_group::{ResponseGroup, ResponseGroupExt};

/// The group of an item, `None` if it can't be grouped.
pub type GroupFn<T> = dyn Fn(&mut T) -> Option<&mut Option<String>>;

pub struct SelectableList<T> {
    items: Vec<SelectableEntry<T>>,
    last_selected: Option<usize>,
}

impl<T> SelectableList<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            last_selected: None,
        }
    }

    /// Shows the items (topmost first), with items in the same group (from
    /// `group_of`, `None` for items that can't be grouped) shown together
    /// under a collapsible header that selects, hides and shows them all.
    /// Returns a group the user asked to save as one image (its items).
    pub fn show_grouped(&mut self, ui: &mut Ui, group_of: &GroupFn<T>) -> Option<Vec<usize>> {
        let n = self.items.len();
        let groups: Vec<Option<String>> =
            self.items.iter_mut().map(|it| group_of(&mut it.inner).and_then(|g| g.clone())).collect();
        let names: Vec<String> = groups.iter().flatten().unique().cloned().collect();
        let mut done: Vec<&str> = Vec::new();
        let mut export = None;
        for i in (0..n).rev() {
            let Some(g) = groups[i].as_deref() else {
                self.show_item(ui, i, Some(group_of));
                continue;
            };
            if done.contains(&g) {
                continue;
            }
            done.push(g);
            // Members top to bottom, shown where the topmost one is.
            let members: Vec<usize> = (0..n).rev().filter(|&j| groups[j].as_deref() == Some(g)).collect();
            let (open, ex) = self.group_header(ui, g, &members, &names, group_of);
            if ex {
                export = Some(members.clone());
            }
            if open {
                ui.indent(("layer_group", g), |ui| {
                    for &j in &members {
                        self.show_item(ui, j, Some(group_of));
                    }
                });
            }
        }
        export
    }

    /// Hide or show every item `which` picks.
    pub fn set_hidden_where(&mut self, which: impl Fn(&T) -> bool, hidden: bool) {
        for i in 0..self.items.len() {
            if which(&self.items[i].inner) {
                self.set_hidden(i, hidden);
            }
        }
    }
    /// Hide or show the selected items.
    pub fn set_selected_hidden(&mut self, hidden: bool) {
        for i in self.iter_selected_indexes().collect_vec() {
            self.set_hidden(i, hidden);
        }
    }
    /// Whether any item `which` picks is hidden / shown: (any hidden, any shown).
    pub fn hidden_state_where(&mut self, which: impl Fn(&T) -> bool) -> (bool, bool) {
        let (mut any_hidden, mut any_shown) = (false, false);
        for i in 0..self.items.len() {
            if which(&self.items[i].inner) {
                if self.is_hidden(i) {
                    any_hidden = true;
                } else {
                    any_shown = true;
                }
            }
        }
        (any_hidden, any_shown)
    }

    fn set_hidden(&mut self, i: usize, hidden: bool) {
        let item = &mut self.items[i];
        *(item.hidden_fn)(&mut item.inner) = hidden;
    }
    fn is_hidden(&mut self, i: usize) -> bool {
        let item = &mut self.items[i];
        *(item.hidden_fn)(&mut item.inner)
    }

    /// Header row of a group. Returns whether it is open, and whether the
    /// user asked to save it as one image.
    /// `names`: every group's name (to merge this one into another).
    fn group_header(
        &mut self,
        ui: &mut Ui,
        name: &str,
        members: &[usize],
        names: &[String],
        group_of: &GroupFn<T>,
    ) -> (bool, bool) {
        let id = ui.make_persistent_id(("layer_group_open", name));
        let mut open = ui.data_mut(|d| *d.get_persisted_mut_or(id, false));
        let all_hidden = members.iter().all(|&j| self.is_hidden(j));
        let all_selected = members.iter().all(|&j| self.items[j].selected);
        let mut export = false;
        ui.horizontal(|ui| {
            let arrow = if open { "\u{25bc}" } else { "\u{25b6}" };
            if ui.add(egui::Button::new(arrow).frame(false)).on_hover_text("Show / hide the list").clicked() {
                open = !open;
            }
            let resp = ui
                .selectable_label(all_selected, format!("{name} ({})", members.len()))
                .on_hover_text("Click to select the whole group; right-click for more");
            if resp.clicked() {
                let ctrl = ui.input(|i| i.modifiers.ctrl);
                if !ctrl {
                    self.clear_selected();
                }
                let select = !(ctrl && all_selected);
                for &j in members {
                    self.items[j].selected = select;
                }
            }
            if resp.double_clicked() {
                open = !open;
            }
            resp.context_menu(|ui| {
                let mut new_name = name.to_string();
                ui.horizontal(|ui| {
                    ui.label("Name");
                    if ui.text_edit_singleline(&mut new_name).changed() && !new_name.trim().is_empty() {
                        for &j in members {
                            if let Some(g) = group_of(&mut self.items[j].inner) {
                                *g = Some(new_name.clone());
                            }
                        }
                    }
                });
                let others: Vec<&String> = names.iter().filter(|n| n.as_str() != name).collect();
                ui.add_enabled_ui(!others.is_empty(), |ui| {
                    ui.menu_button("Merge into", |ui| {
                        for other in others {
                            if ui
                                .button(other.as_str())
                                .on_hover_text(format!("Move this group's stamps into \u{201c}{other}\u{201d}"))
                                .clicked()
                            {
                                for &j in members {
                                    if let Some(g) = group_of(&mut self.items[j].inner) {
                                        *g = Some(other.clone());
                                    }
                                }
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_disabled_hover_text("There is no other group to merge into");
                });
                if ui.button("Polish & save as one image\u{2026}").clicked() {
                    export = true;
                    ui.close();
                }
                if ui.button(if all_hidden { "Show all" } else { "Hide all" }).clicked() {
                    for &j in members {
                        self.set_hidden(j, !all_hidden);
                    }
                    ui.close();
                }
                if ui.button("Ungroup").clicked() {
                    for &j in members {
                        if let Some(g) = group_of(&mut self.items[j].inner) {
                            *g = None;
                        }
                    }
                    ui.close();
                }
            });
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                let mut hidden = all_hidden;
                if hidden_toggle_icon(ui, &mut hidden).clicked() {
                    for &j in members {
                        self.set_hidden(j, hidden);
                    }
                }
            });
        });
        ui.data_mut(|d| d.insert_persisted(id, open));
        (open, export)
    }

    fn show_item(&mut self, ui: &mut Ui, i: usize, group_of: Option<&GroupFn<T>>) {
        let (resp, eye) = list_item(ui, &mut self.items[i]);
        if eye && self.items[i].selected {
            // The eye of a selected item hides or shows the whole selection.
            let hidden = self.is_hidden(i);
            self.set_selected_hidden(hidden);
        }
        let mut resp = resp.synchronize(&mut self.items[i].resp_group);
        if resp.sync.hovered() {
            resp.orig = resp.orig.highlight();
        }
        if resp.orig.clicked() {
            if ui.input(|i| i.modifiers.shift) && self.last_selected.is_some() {
                let mut i = i as isize;
                let last = self.last_selected.unwrap() as isize;
                let add = ((i < last) as isize) * 2 - 1;
                while i != last {
                    self.items[i as usize].selected = true;
                    i += add;
                }
            } else if ui.input(|i| i.modifiers.ctrl) {
                self[i].selected = !self[i].selected;
            } else {
                self.clear_selected();
                self[i].selected = true;
            }
            if self[i].selected {
                self.last_selected = Some(i);
            }
        }
        resp.orig.context_menu(|ui| {
            if ui.button("Hide selected").clicked() {
                for n in self.iter_selected_indexes().collect_vec() {
                    self.set_hidden(n, true);
                }
            }
            if ui.button("Show selected").clicked() {
                for n in self.iter_selected_indexes().collect_vec() {
                    self.set_hidden(n, false);
                }
            }
            let Some(group_of) = group_of else { return };
            let selected = self.iter_selected_indexes().collect_vec();
            let groupable = |s: &mut Self, n: usize| group_of(&mut s.items[n].inner).is_some();
            let can_group: Vec<usize> = selected.iter().copied().filter(|&n| groupable(self, n)).collect();
            ui.separator();
            if ui
                .add_enabled(can_group.len() >= 2, egui::Button::new("Group selected"))
                .on_disabled_hover_text("Select two or more stamps")
                .clicked()
            {
                // First free "Group N".
                let names: Vec<String> = (0..self.items.len())
                    .filter_map(|n| group_of(&mut self.items[n].inner).and_then(|g| g.clone()))
                    .collect();
                let name = (1..)
                    .map(|k| format!("Group {k}"))
                    .find(|c| !names.contains(c))
                    .unwrap_or_default();
                for &n in &can_group {
                    if let Some(g) = group_of(&mut self.items[n].inner) {
                        *g = Some(name.clone());
                    }
                }
                ui.close();
            }
            let in_group = can_group
                .iter()
                .any(|&n| group_of(&mut self.items[n].inner).is_some_and(|g| g.is_some()));
            if ui.add_enabled(in_group, egui::Button::new("Remove from group")).clicked() {
                for &n in &can_group {
                    if let Some(g) = group_of(&mut self.items[n].inner) {
                        *g = None;
                    }
                }
                ui.close();
            }
        });
    }
    pub fn get_hovered(&self, ctx: &Context) -> Option<&SelectableEntry<T>> {
        self.items
            .iter()
            .find_map(|item| item.resp_group.response(ctx)?.hovered().then_some(item))
    }
    pub fn clear_selected(&mut self) {
        for item in &mut self.items {
            item.selected = false;
        }
        self.last_selected = None;
    }
    pub fn iter_selected_indexes<'a>(&'a self) -> impl DoubleEndedIterator<Item = usize> + 'a {
        self.iter()
            .enumerate()
            .filter_map(|(i, item)| item.selected.then_some(i))
    }
    pub fn iter_selected(&self) -> impl Iterator<Item = &SelectableEntry<T>> {
        self.iter().filter(|item| item.selected)
    }
    pub fn move_indexes_up(&mut self, indexes: &[usize]) -> Vec<usize> {
        let mut moved = Vec::new();
        for (i, index) in indexes.iter().copied().enumerate() {
            if i == index {
                continue;
            }
            self.swap(index, index - 1);
            moved.push(index - 1);
        }
        moved
    }
    pub fn move_indexes_down(&mut self, indexes: &[usize]) -> Vec<usize> {
        let mut moved = Vec::new();
        for (i, index) in indexes.iter().copied().rev().enumerate() {
            let i = self.len() - i - 1;
            if i == index {
                continue;
            }
            self.swap(index, index + 1);
            moved.push(index + 1);
        }
        moved.reverse();
        moved
    }
}
impl<T> Deref for SelectableList<T> {
    type Target = Vec<SelectableEntry<T>>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}
impl<T> DerefMut for SelectableList<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}

pub struct SelectableEntry<T> {
    id: egui::Id,
    pub inner: T,
    pub selected: bool,
    pub resp_group: ResponseGroup,
    construct_fn: Box<dyn Fn(&T) -> Atoms>,
    hidden_fn: Box<dyn Fn(&mut T) -> &mut bool>,
}
impl<T> SelectableEntry<T> {
    pub fn new(
        id_salt: impl Hash,
        data: T,
        construct_fn: impl Fn(&T) -> Atoms + 'static,
        hidden_fn: impl Fn(&mut T) -> &mut bool + 'static,
    ) -> Self {
        Self {
            id: egui::Id::new(id_salt),
            inner: data,
            selected: false,
            construct_fn: Box::new(construct_fn),
            resp_group: ResponseGroup::new(),
            hidden_fn: Box::new(hidden_fn),
        }
    }
}
impl<T> Deref for SelectableEntry<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T> DerefMut for SelectableEntry<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// The row, and whether its eye was clicked.
fn list_item<'a, T>(ui: &mut Ui, item: &'a mut SelectableEntry<T>) -> (Response, bool) {
    let mut eye = false;
    let (response, hidden) = {
        let inner = &mut item.inner;
        let mut hidden = *(item.hidden_fn)(inner);
        let atoms = (item.construct_fn)(&item.inner);
        let id = item.id;
        let font_selection = FontSelection::Style(TextStyle::Button);
        let font_id = font_selection.resolve(ui.style());
        let mut layout = AtomLayout::new(atoms)
            .id(id)
            .sense(Sense::click())
            .wrap_mode(egui::TextWrapMode::Extend)
            .fallback_font(FontSelection::FontId(font_id.clone()));

        let selected = item.selected;
        let min_size = egui::Vec2::new(0., ui.spacing().interact_size.y);

        layout.map_atoms(|atom| {
            if matches!(&atom.kind, AtomKind::Image(_)) {
                let height = ui.fonts(|f| f.row_height(&font_id));
                atom.atom_max_height(height)
            } else {
                atom
            }
        });

        let button_padding = ui.spacing().button_padding;
        let mut prepared = ui
            .horizontal(|ui| {
                let prepared = layout
                    .frame(Frame::new().inner_margin(button_padding))
                    .min_size(min_size)
                    .allocate(ui);
                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    eye = hidden_toggle_icon(ui, &mut hidden).clicked();
                });
                prepared
            })
            .inner;

        let response = if ui.is_rect_visible(prepared.response.rect) {
            let mut visuals = ui.style().interact_selectable(&prepared.response, selected);
            if hidden {
                visuals.fg_stroke.color = visuals.fg_stroke.color.gamma_multiply(0.3);
            }

            prepared.map_images(|image| image.tint(visuals.text_color()));

            prepared.fallback_text_color = visuals.text_color();

            prepared.frame = prepared
                .frame
                .inner_margin(
                    button_padding + egui::Vec2::splat(visuals.expansion)
                        - egui::Vec2::splat(visuals.bg_stroke.width),
                )
                .outer_margin(-egui::Vec2::splat(visuals.expansion))
                .stroke(visuals.bg_stroke)
                .corner_radius(visuals.corner_radius);
            if selected {
                prepared.frame = prepared.frame.fill(visuals.weak_bg_fill)
            }

            prepared.paint(ui)
        } else {
            AtomLayoutResponse::empty(prepared.response)
        }
        .response;
        (response, hidden)
    };
    *(item.hidden_fn)(&mut item.inner) = hidden;
    (response, eye)
}

pub fn hidden_toggle_icon(ui: &mut Ui, hidden: &mut bool) -> Response {
    let source = if *hidden {
        egui::include_image!("../../assets/hide_icon.png")
    } else {
        egui::include_image!("../../assets/show_icon.png")
    };

    let size = egui::vec2(ui.spacing().interact_size.y, ui.spacing().interact_size.y);

    let image = egui::Image::new(source).fit_to_exact_size(size);

    let mut response = ui.allocate_response(size, Sense::CLICK | Sense::DRAG);

    let mut visuals = *ui.style().interact(&response);
    if *hidden {
        visuals.fg_stroke.color = visuals.fg_stroke.color.gamma_multiply(0.3);
    }

    let tint = visuals.text_color();
    response = ui.put(
        response.rect,
        egui::Button::image(image.tint(tint)).frame(false),
    );

    if response.clicked() {
        *hidden = !*hidden;
    }

    response.on_hover_text(if *hidden { "Show" } else { "Hide" })
}
