use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use parking_lot::{RwLock, RwLockReadGuard};

pub struct SharedState<T> {
    inner: Arc<RwLock<T>>,
    shared_epoch: Arc<AtomicUsize>,
    local_epoch: usize,
}
impl<T> SharedState<T> {
    pub fn new(init: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(init)),
            shared_epoch: Arc::new(AtomicUsize::new(0)),
            local_epoch: 0,
        }
    }
    pub fn new_default() -> Self
    where
        T: Default,
    {
        Self::new(Default::default())
    }
    pub fn peek(&self) -> RwLockReadGuard<'_, T> {
        self.inner.read()
    }
    pub fn read(&mut self) -> RwLockReadGuard<'_, T> {
        self.local_epoch = self.shared_epoch.load(Ordering::Relaxed);
        self.inner.read()
    }
    pub fn read_new(&mut self) -> Option<RwLockReadGuard<'_, T>> {
        let shared_epoch = self.shared_epoch.load(Ordering::Relaxed);
        (self.local_epoch < shared_epoch).then(|| {
            self.local_epoch = shared_epoch;
            self.inner.read()
        })
    }
    pub fn write(&mut self, val: T) {
        self.modify(|state| *state = val);
    }
    pub fn modify(&mut self, mod_fn: impl FnOnce(&mut T)) {
        mod_fn(&mut self.inner.write());
        // Data race here with multiple writers
        let shared_epoch = self.shared_epoch.fetch_add(1, Ordering::Relaxed);
        self.local_epoch = shared_epoch + 1;
    }
    pub fn modify_conditional(
        &mut self,
        check_fn: impl FnOnce(&T) -> bool,
        mod_fn: impl FnOnce(&mut T),
    ) -> bool {
        let mut lock = self.inner.upgradable_read();
        let wants_modify = check_fn(&lock);
        if wants_modify {
            lock.with_upgraded(mod_fn);
            let shared_epoch = self.shared_epoch.fetch_add(1, Ordering::Relaxed);
            self.local_epoch = shared_epoch + 1;
        }
        wants_modify
    }
}
impl<T> Clone for SharedState<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            shared_epoch: self.shared_epoch.clone(),
            local_epoch: self.local_epoch.clone(),
        }
    }
}
