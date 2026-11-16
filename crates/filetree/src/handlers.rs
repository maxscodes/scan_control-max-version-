use std::path::Path;

use notify_typed::Event;

pub trait LoadHandler {
    type FileData;
    type FolderData;
    fn load_file(&mut self, path: &Path) -> Option<Self::FileData>;
    fn load_folder(&mut self, path: &Path) -> Option<Self::FolderData>;
}
pub trait UpdateHandler: LoadHandler {
    fn rename_file(&mut self, _old: &Path, _new: &Path, _data: &mut Self::FileData) {}
    fn rename_folder(&mut self, _old: &Path, _new: &Path, _data: &mut Self::FolderData) {}
    fn delete_file(&mut self, _path: &Path, _data: Self::FileData) {}
    fn delete_folder(&mut self, _path: &Path, _data: Self::FolderData) {}
}
impl LoadHandler for () {
    type FileData = ();
    type FolderData = ();
    fn load_file(&mut self, _path: &Path) -> Option<Self::FileData> {
        Some(())
    }
    fn load_folder(&mut self, _path: &Path) -> Option<Self::FolderData> {
        Some(())
    }
}

impl<T, F: FnMut(&Path) -> Option<T>> LoadHandler for F {
    type FileData = T;
    type FolderData = ();
    fn load_file(&mut self, path: &Path) -> Option<Self::FileData> {
        (self)(path)
    }
    fn load_folder(&mut self, _path: &Path) -> Option<Self::FolderData> {
        Some(())
    }
}

impl<H: LoadHandler, F: FnMut(Event)> LoadHandler for (H, F) {
    type FileData = H::FileData;
    type FolderData = H::FolderData;
    fn load_file(&mut self, path: &Path) -> Option<Self::FileData> {
        (self.1)(Event::Create {
            path: path.to_path_buf(),
        });
        H::load_file(&mut self.0, path)
    }
    fn load_folder(&mut self, path: &Path) -> Option<Self::FolderData> {
        (self.1)(Event::Create {
            path: path.to_path_buf(),
        });
        H::load_folder(&mut self.0, path)
    }
}
impl<H: LoadHandler, F: FnMut(Event)> UpdateHandler for (H, F) {
    fn rename_file(&mut self, old: &Path, new: &Path, _data: &mut Self::FileData) {
        (self.1)(Event::Rename {
            from: old.to_path_buf(),
            to: new.to_path_buf(),
        })
    }
    fn rename_folder(&mut self, old: &Path, new: &Path, _data: &mut Self::FolderData) {
        (self.1)(Event::Rename {
            from: old.to_path_buf(),
            to: new.to_path_buf(),
        })
    }
    fn delete_file(&mut self, path: &Path, _data: Self::FileData) {
        (self.1)(Event::Delete {
            path: path.to_path_buf(),
        })
    }
    fn delete_folder(&mut self, path: &Path, _data: Self::FolderData) {
        (self.1)(Event::Delete {
            path: path.to_path_buf(),
        })
    }
}
