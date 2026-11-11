use std::path::PathBuf;

use glam::DAffine2;

use crate::{project::ProjectDb, scan_view::ImageEncoder, view_object};

pub struct ObjectImportDialog {
    channel_tx: crossbeam::channel::Sender<view_object::Object>,
    channel_rx: crossbeam::channel::Receiver<view_object::Object>,
    dialog: rfd::FileDialog,
}
impl ObjectImportDialog {
    pub fn new() -> Self {
        let dialog = rfd::FileDialog::new()
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
        let dialog = rfd::FileDialog::new()
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
        let dialog = rfd::FileDialog::new()
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
