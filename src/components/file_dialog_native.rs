use std::{path::PathBuf, sync::OnceLock};

use glam::DAffine2;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle,
};

use crate::{project::ProjectDb, scan_view::ImageEncoder, view_object};

/// The main window, used as the owner of every file dialog. Without an
/// owner, Windows may open the dialog behind the app window, so it looks like
/// the button did nothing.
struct MainWindow {
    window: RawWindowHandle,
    display: RawDisplayHandle,
}
// Safety: the handles are only identifiers (an HWND on Windows); the main
// window outlives every dialog.
unsafe impl Send for MainWindow {}
unsafe impl Sync for MainWindow {}
impl HasWindowHandle for MainWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        // Safety: the main window lives as long as the app.
        Ok(unsafe { WindowHandle::borrow_raw(self.window) })
    }
}
impl HasDisplayHandle for MainWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // Safety: as above.
        Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
    }
}
static MAIN_WINDOW: OnceLock<MainWindow> = OnceLock::new();

/// Remember the main window so [`dialog`] can attach dialogs to it. Call once
/// at startup, before any dialog is created.
pub fn set_main_window(w: &(impl HasWindowHandle + HasDisplayHandle)) {
    match (w.window_handle(), w.display_handle()) {
        (Ok(window), Ok(display)) => {
            let _ = MAIN_WINDOW.set(MainWindow {
                window: window.as_raw(),
                display: display.as_raw(),
            });
        }
        (Err(e), _) | (_, Err(e)) => tracing::warn!("no window handle for file dialogs: {e}"),
    }
}

/// A file dialog owned by the main window, so it opens in front of it.
pub fn dialog() -> rfd::FileDialog {
    let d = rfd::FileDialog::new();
    match MAIN_WINDOW.get() {
        Some(w) => d.set_parent(w),
        None => d,
    }
}

pub struct ObjectImportDialog {
    channel_tx: crossbeam::channel::Sender<view_object::Object>,
    channel_rx: crossbeam::channel::Receiver<view_object::Object>,
    dialog: rfd::FileDialog,
}
impl ObjectImportDialog {
    pub fn new() -> Self {
        let dialog = dialog()
            .add_filter("Any", &["png", "jpeg", "gds"])
            .add_filter("Image", &["png", "jpeg"])
            .add_filter("GDS", &["gds"])
            .set_title("Import Files");
        let (channel_tx, channel_rx) = crossbeam::channel::unbounded();
        Self {
            channel_tx,
            channel_rx,
            dialog,
        }
    }
    pub fn pick_files(&mut self, encoder: ImageEncoder, transform: DAffine2) {
        let dialog = self.dialog.clone();
        let channel = self.channel_tx.clone();
        std::thread::spawn(move || {
            if let Some(paths) = dialog.pick_files() {
                for path in paths {
                    if let Some(object) = view_object::Object::import(path, &encoder, transform) {
                        channel.send(object).unwrap();
                    }
                }
            }
        });
    }
    pub fn try_recv_object(&mut self) -> Option<view_object::Object> {
        match self.channel_rx.try_recv() {
            Ok(path) => Some(path),
            Err(_) => None,
        }
    }
}

pub struct ProjectOpenDialog {
    channel_tx: crossbeam::channel::Sender<ProjectDb>,
    channel_rx: crossbeam::channel::Receiver<ProjectDb>,
    dialog: rfd::FileDialog,
}
impl ProjectOpenDialog {
    pub fn new() -> Self {
        let dialog = dialog()
            .add_filter("Project File", &["scp"])
            .set_title("Open Project");
        let (channel_tx, channel_rx) = crossbeam::channel::unbounded();
        Self {
            channel_tx,
            channel_rx,
            dialog,
        }
    }
    pub fn pick_file(&mut self) {
        let dialog = self.dialog.clone();
        let channel = self.channel_tx.clone();
        std::thread::spawn(move || {
            if let Some(path) = dialog.pick_file() {
                match ProjectDb::open(path.clone()) {
                    Ok(project) => {
                        let _ = channel.send(project);
                    }
                    Err(e) => tracing::error!("could not open {}: {e}", path.display()),
                }
            }
        });
    }
    pub fn try_recv_project(&mut self) -> Option<ProjectDb> {
        match self.channel_rx.try_recv() {
            Ok(project) => Some(project),
            Err(_) => None,
        }
    }
}

pub struct ProjectSaveDialog {
    channel_tx: crossbeam::channel::Sender<PathBuf>,
    channel_rx: crossbeam::channel::Receiver<PathBuf>,
    dialog: rfd::FileDialog,
}
impl ProjectSaveDialog {
    pub fn new() -> Self {
        let dialog = dialog()
            .set_title("Save Project")
            .set_can_create_directories(true)
            .set_file_name("project.scp");
        let (channel_tx, channel_rx) = crossbeam::channel::unbounded();
        Self {
            channel_tx,
            channel_rx,
            dialog,
        }
    }
    pub fn select_path(&mut self) {
        let dialog = self.dialog.clone();
        let channel = self.channel_tx.clone();
        std::thread::spawn(move || {
            if let Some(mut path) = dialog.save_file() {
                path.set_extension("scp");
                channel.send(path).unwrap();
            }
        });
    }
    pub fn try_recv_path(&mut self) -> Option<PathBuf> {
        match self.channel_rx.try_recv() {
            Ok(path) => Some(path),
            Err(_) => None,
        }
    }
}
