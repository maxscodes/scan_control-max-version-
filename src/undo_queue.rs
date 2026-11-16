use std::any::Any;

pub struct UndoQueue<S> {
    queue: Vec<Box<dyn StateModify<S>>>,
    index: usize,
}
impl<S: 'static> UndoQueue<S> {
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
            index: 0,
        }
    }
    pub fn push<T: StateModify<S>>(&mut self, app_state: &mut S, mut modifier: T) {
        if !modifier.redo(app_state) {
            return;
        }
        self.queue.truncate(self.index);
        if let Some(prev) = self.queue.last_mut() {
            if modifier.combine(prev.as_mut()) {
                return;
            }
        }
        self.queue.push(Box::new(modifier));
        self.index += 1;
    }
    pub fn can_undo(&self) -> bool {
        self.index > 0
    }
    pub fn can_redo(&self) -> bool {
        self.index < self.queue.len()
    }
    pub fn undo(&mut self, app_state: &mut S) {
        if self.index == 0 {
            return;
        }
        self.index -= 1;
        let entry = &mut self.queue[self.index];
        entry.undo(app_state);
    }
    pub fn redo(&mut self, app_state: &mut S) {
        if self.index == self.queue.len() {
            return;
        }
        let entry = &mut self.queue[self.index];
        entry.redo(app_state);
        self.index += 1;
    }
}

pub trait StateModify<S>: Any {
    fn redo(&mut self, state: &mut S) -> bool;
    fn undo(&mut self, state: &mut S);
    fn combine(&mut self, _previous: &mut dyn Any) -> bool {
        false
    }
}

impl<S, F> StateModify<S> for F
where
    F: Fn(&mut S) -> bool + 'static,
{
    fn redo(&mut self, state: &mut S) -> bool {
        self(state)
    }

    fn undo(&mut self, state: &mut S) {
        self(state);
    }
}

impl<S, F1, F2> StateModify<S> for (F1, F2)
where
    F1: Fn(&mut S) -> bool + 'static,
    F2: Fn(&mut S) + 'static,
{
    fn redo(&mut self, state: &mut S) -> bool {
        self.0(state)
    }

    fn undo(&mut self, state: &mut S) {
        self.1(state);
    }
}
