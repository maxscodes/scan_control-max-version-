use std::sync::Arc;

#[derive(Default)]
pub struct ChannelState {
    names: Arc<Box<[String]>>,
    opts: Vec<usize>,
    pub selection: Option<usize>,
}
impl ChannelState {
    pub fn modify_opts(&mut self, mod_fn: impl FnOnce(&mut Vec<usize>)) {
        mod_fn(&mut self.opts);
        if self
            .selection
            .is_none_or(|selection| !self.opts.contains(&selection))
        {
            // Prefer the topography channel (Z) when it is recorded.
            let z = self.opts.iter().copied().find(|&i| {
                self.names
                    .get(i)
                    .is_some_and(|n| n.trim_start().starts_with('Z'))
            });
            self.selection = z.or(self.opts.first().copied());
        }
    }
    pub fn options(&self) -> &[usize] {
        &self.opts
    }
    pub fn write_signal_names(&mut self, names: Arc<Box<[String]>>) {
        self.names = names;
    }
    pub fn signal_names(&self) -> &[String] {
        &self.names
    }
    pub fn channel_opts_names<'s>(&'s self) -> impl Iterator<Item = String> + 's {
        self.opts.iter().map(|&opt| self.name(opt))
    }
    pub fn selected_as_string(&self) -> Option<String> {
        self.selection.map(|sel| self.name(sel))
    }
    /// Name of a signal index, or a placeholder if the name list is stale.
    fn name(&self, idx: usize) -> String {
        self.names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("Signal {idx}"))
    }
    pub fn set_selection_by_name(&mut self, name: &str) {
        self.selection = self
            .opts
            .iter()
            .copied()
            .find(|&i| self.name(i) == name);
    }
    pub fn unit(&self) -> Option<String> {
        self.selected_as_string()
            .and_then(|sel| sel.split(['(', ')']).nth(1).map(str::to_string))
    }
}
