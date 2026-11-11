use std::{
    env::temp_dir,
    error::Error,
    path::{Path, PathBuf},
};

use redb::{Database, WriteTransaction, backends::InMemoryBackend};
use uuid::Uuid;

use crate::scan_view::ImageEncoder;

pub struct ProjectDb {
    path: PathBuf,
    db: Database,
    is_temp: bool,
}
impl ProjectDb {
    pub fn new_temp() -> Result<ProjectDb, redb::DatabaseError> {
        let mut path = temp_dir();
        let name = format!("temp_db_{}.scp", Uuid::new_v4());
        path.push(name);
        let db = Database::builder().create(&path)?;
        Ok(Self {
            db,
            path,
            is_temp: true,
        })
    }
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }
    /// Copy the database file to `path` and continue working on the copy.
    /// The caller must commit the current state to this database first.
    /// On failure the original database is reopened, so nothing is lost.
    pub fn save_as(&mut self, path: PathBuf) -> Result<(), Box<dyn Error>> {
        // Saving onto the file we already have open: the commit was the save.
        if same_file(&self.path, &path) {
            return Ok(());
        }
        let tmp_db = Database::builder().create_with_backend(InMemoryBackend::new())?;
        let mut old_db = std::mem::replace(&mut self.db, tmp_db);
        if let Err(e) = old_db.compact() {
            // Compaction is only an optimisation; keep going without it.
            tracing::warn!("project compact failed: {e}");
        }
        drop(old_db);
        // Copy to a temporary name next to the target, then rename, so a
        // failed copy never leaves a truncated project file behind.
        let partial = path.with_extension("scp.partial");
        let copied = std::fs::copy(&self.path, &partial)
            .map_err(Box::<dyn Error>::from)
            .and_then(|_| {
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
                std::fs::rename(&partial, &path)?;
                Ok(())
            });
        if let Err(e) = copied {
            let _ = std::fs::remove_file(&partial);
            self.db = Database::open(&self.path)?;
            return Err(e);
        }
        match Self::open(path) {
            Ok(new) => {
                *self = new;
                Ok(())
            }
            Err(e) => {
                self.db = Database::open(&self.path)?;
                Err(e)
            }
        }
    }
    pub fn open(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let new_db = Database::open(path.clone())?;
        Ok(Self {
            db: new_db,
            path: path,
            is_temp: false,
        })
    }
    pub fn current_path(&self) -> &Path {
        &self.path
    }
    pub fn db(&self) -> &Database {
        &self.db
    }
    pub fn update_from(&self, val: &impl Persistant) -> Result<(), Box<dyn Error>> {
        let txn = self.db.begin_write()?;
        val.db_update(&txn)?;
        Ok(())
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

pub trait Persistant: Sized {
    fn db_update<'t>(&self, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>>;
    fn db_remove<'t>(id: Uuid, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>>;
    fn db_insert<'t>(&self, txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>>;
    fn db_read<'t>(
        id: Uuid,
        txn: &'t WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn Error>>;
    fn db_dump_stats<'t>(txn: &'t WriteTransaction) -> Result<(), Box<dyn Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use redb::{ReadableDatabase, ReadableTableMetadata};

    const T: redb::TableDefinition<u64, u64> = redb::TableDefinition::new("t");

    fn write(p: &ProjectDb, n: u64) {
        let txn = p.db().begin_write().unwrap();
        txn.open_table(T).unwrap().insert(n, n).unwrap();
        txn.commit().unwrap();
    }
    fn count(p: &ProjectDb) -> u64 {
        p.db().begin_read().unwrap().open_table(T).unwrap().len().unwrap()
    }

    #[test]
    fn save_as_round_trip() {
        let dir = temp_dir().join(format!("scp_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.scp");
        let b = dir.join("b.scp");

        let mut p = ProjectDb::new_temp().unwrap();
        write(&p, 1);
        p.save_as(a.clone()).unwrap();
        assert!(!p.is_temp());
        assert_eq!(count(&p), 1);

        // Save onto the open file is a no-op, data stays.
        write(&p, 2);
        p.save_as(a.clone()).unwrap();
        assert_eq!(count(&p), 2);

        // Save As to a new file, then overwrite an existing file.
        p.save_as(b.clone()).unwrap();
        write(&p, 3);
        p.save_as(a.clone()).unwrap();
        assert_eq!(count(&p), 3);
        drop(p);

        let reopened = ProjectDb::open(a).unwrap();
        assert_eq!(count(&reopened), 3);
        let other = ProjectDb::open(b).unwrap();
        assert_eq!(count(&other), 3); // work continued in b after Save As
        drop((reopened, other));
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Lists what each project file holds. Run with
/// `PROJECT_FILES="a.scp;b.scp" cargo test --bin scan_control inspect_projects -- --ignored --nocapture`.
#[cfg(test)]
mod inspect {
    use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
    use uuid::Uuid;

    #[test]
    #[ignore]
    fn inspect_projects() {
        const LIST: TableDefinition<Uuid, u64> = TableDefinition::new("object_list_table_v1");
        const TYPES: TableDefinition<Uuid, &str> = TableDefinition::new("object_type_table_v1");
        const HIST: TableDefinition<Uuid, Vec<[f64; 2]>> =
            TableDefinition::new("scanarea_history_table_v1");
        let files = std::env::var("PROJECT_FILES").unwrap_or_default();
        for f in files.split(';').filter(|s| !s.is_empty()) {
            let db = match Database::open(f) {
                Ok(db) => db,
                Err(e) => {
                    println!("{f}: cannot open ({e})");
                    continue;
                }
            };
            let txn = db.begin_read().unwrap();
            let n = txn.open_table(LIST).map(|t| t.len().unwrap()).unwrap_or(0);
            let mut kinds = Vec::new();
            if let Ok(t) = txn.open_table(TYPES) {
                for e in t.iter().unwrap() {
                    kinds.push(e.unwrap().1.value().to_string());
                }
            }
            let hist: usize = txn
                .open_table(HIST)
                .map(|t| t.iter().unwrap().map(|e| e.unwrap().1.value().len()).sum())
                .unwrap_or(0);
            kinds.sort();
            println!("{f}: {n} objects {kinds:?}, {hist} history points");
        }
    }
}
