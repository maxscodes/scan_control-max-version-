use std::{ffi::OsStr, path::PathBuf};

use egui::{Atoms, Image, IntoAtoms, Ui};
use glam::{DAffine2, DVec2};
use redb::{ReadableTable, ReadableTableMetadata, TableDefinition};
use tracing::error;
use uuid::Uuid;

use crate::{
    components::selectable_list::SelectableEntry,
    connection::{LiveImage, ScanArea},
    project::Persistant,
    scan_view::{FileImage, GDSImage, ImageEncoder},
};

/// Where an object sits on the map, to undo moving it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    Transform(DAffine2),
    /// An imported image's four corners.
    Corners([DVec2; 4]),
}

pub enum Object {
    Gds {
        image: GDSImage,
        path: PathBuf,
        hidden: bool,
    },
    File {
        image: FileImage,
        path: PathBuf,
        hidden: bool,
    },
    ScanImage {
        image: LiveImage,
        name: String,
        hidden: bool,
        /// Layer group (e.g. the tiles of one survey), shown together in
        /// Layers and hidden or shown together.
        group: Option<String>,
    },
    ScanArea {
        image: ScanArea,
        hidden: bool,
    },
}

impl Object {
    /// This object as an entry of the layer list.
    pub fn into_entry(self, id_salt: impl std::hash::Hash) -> SelectableEntry<Object> {
        SelectableEntry::new(id_salt, self, |o| o.list_atoms(), |o| o.hidden_mut())
    }
    /// Where the object is on the map. `None` for the scan region, which
    /// follows the instrument (coarse moves) rather than being placed.
    pub fn placement(&self) -> Option<Placement> {
        match self {
            Object::Gds { image, .. } => Some(Placement::Transform(image.transform)),
            Object::File { image, .. } => Some(Placement::Corners(image.world_points)),
            Object::ScanImage { image, .. } => Some(Placement::Transform(image.transform)),
            Object::ScanArea { .. } => None,
        }
    }
    pub fn set_placement(&mut self, p: Placement) {
        match (self, p) {
            (Object::Gds { image, .. }, Placement::Transform(t)) => image.transform = t,
            (Object::ScanImage { image, .. }, Placement::Transform(t)) => image.transform = t,
            (Object::File { image, .. }, Placement::Corners(c)) => image.set_world_points(c),
            _ => {}
        }
    }
    pub fn import(path: PathBuf, encoder: &ImageEncoder, transform: DAffine2) -> Option<Self> {
        match path.extension().and_then(|os| os.to_str()) {
            Some("gds") | Some("GDS") => {
                let image = GDSImage::new_from_file(
                    encoder,
                    &path,
                    DAffine2::from_angle_translation(0., transform.translation),
                );
                Some(Self::Gds {
                    image,
                    path,
                    hidden: false,
                })
            }
            Some("png") | Some("jpeg") | Some("PNG") | Some("JPEG") => {
                let image = FileImage::new(encoder, &path, transform.into());
                Some(Self::File {
                    image,
                    path,
                    hidden: false,
                })
            }
            Some(_) => {
                error!("tried to import invalid file type: {}", path.display());
                None
            }
            None => {
                error!(
                    "tried to import file with invalid extension: {}",
                    path.display()
                );
                None
            }
        }
    }
    pub fn uuid(&self) -> Uuid {
        match self {
            Object::Gds { image, .. } => image.uuid(),
            Object::File { image, .. } => image.uuid(),
            Object::ScanImage { image, .. } => image.uuid(),
            Object::ScanArea { image, .. } => image.uuid(),
        }
    }
    pub fn list_atoms<'a>(&'a self) -> Atoms<'a> {
        let name = self.name();
        let image = match self {
            Object::Gds { .. } => Image::new(egui::include_image!("../assets/gds_file_icon.png")),
            Object::File { .. } => {
                Image::new(egui::include_image!("../assets/file_image_icon.png"))
            }
            Object::ScanImage { .. } => {
                Image::new(egui::include_image!("../assets/scan_image_icon.png"))
            }
            Object::ScanArea { .. } => {
                Image::new(egui::include_image!("../assets/scan_area_icon.png"))
            }
        };
        (image, name).into_atoms()
    }
    /// The layer group, for objects that can be grouped (stamps).
    pub fn group_mut(&mut self) -> Option<&mut Option<String>> {
        match self {
            Object::ScanImage { group, .. } => Some(group),
            _ => None,
        }
    }
    pub fn hidden_mut(&mut self) -> &mut bool {
        match self {
            Object::Gds { hidden, .. } => hidden,
            Object::File { hidden, .. } => hidden,
            Object::ScanImage { hidden, .. } => hidden,
            Object::ScanArea { hidden, .. } => hidden,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            Object::Gds { path, .. } => path.file_stem().and_then(|os| os.to_str()).unwrap(),
            Object::File { path, .. } => path.file_stem().and_then(|os| os.to_str()).unwrap(),
            Object::ScanImage { name, .. } => name,
            Object::ScanArea { .. } => "Scan Region",
        }
    }
    pub fn is_scalable(&self) -> bool {
        match self {
            Object::Gds { .. } => false,
            Object::File { .. } => true,
            Object::ScanImage { .. } => false,
            Object::ScanArea { .. } => false,
        }
    }
    pub fn as_scan_area(&self) -> Option<&ScanArea> {
        match self {
            Object::ScanArea { image, .. } => Some(image),
            _ => None,
        }
    }
    pub fn as_scan_area_mut(&mut self) -> Option<&mut ScanArea> {
        match self {
            Object::ScanArea { image, .. } => Some(image),
            _ => None,
        }
    }
    pub fn show(&mut self, ui: &mut Ui) {
        match self {
            Object::Gds { image, .. } => image.show(ui),
            Object::File { image, .. } => image.show(ui),
            Object::ScanImage { image, .. } => {
                image.show_image(ui);
            }
            Object::ScanArea { image, .. } => image.show(ui),
        }
    }
    pub fn show_menu(&mut self, ui: &mut Ui, encoder: &ImageEncoder) {
        match self {
            Object::Gds { image, .. } => image.show_menu(ui),
            Object::File { image, .. } => image.show_menu(ui),
            Object::ScanImage { image, .. } => image.show_menu(ui, encoder),
            Object::ScanArea { image, .. } => image.show_menu(ui, encoder),
        }
    }
    pub fn border_transform(&self) -> Option<DAffine2> {
        match self {
            Object::ScanImage { image, .. } => Some(image.transform),
            Object::ScanArea { image, .. } => {
                Some(image.world_transform * DAffine2::from_scale(image.area_size))
            }
            _ => None,
        }
    }
    pub fn transform_center(&self) -> DVec2 {
        match self {
            Object::Gds { image, .. } => image.transform.translation,
            Object::File { image, .. } => image.center(),
            Object::ScanImage { image, .. } => image.transform.translation,
            Object::ScanArea { image, .. } => image.world_transform.translation,
        }
    }
    pub fn apply_transform(&mut self, tran: DAffine2) {
        match self {
            Object::Gds { image, .. } => image.transform = tran * image.transform,
            Object::File { image, .. } => image.transform_world_points(tran),
            Object::ScanImage { image, .. } => image.transform = tran * image.transform,
            Object::ScanArea { image, .. } => {
                let before = image.world_transform;
                image.world_transform = tran * image.world_transform;
                image.manual_drag |= image.world_transform != before;
            }
        }
    }
    pub fn goto_transform(&self) -> DAffine2 {
        match self {
            Object::Gds { image, .. } => DAffine2::from_scale_angle_translation(
                DVec2::splat(image.scale) * image.transform.to_scale_angle_translation().0.x,
                0.,
                image.transform.translation,
            ),
            Object::ScanImage { image, .. } => {
                image.transform * DAffine2::from_scale(DVec2::new(1., -1.))
            }
            Object::ScanArea { image, .. } => {
                image.world_transform
                    * DAffine2::from_scale(DVec2::splat(
                        (image.area_size.x + image.area_size.y) / 2.,
                    ))
            }
            Object::File { image, .. } => {
                let center = image.center();
                let scale = image
                    .world_points
                    .iter()
                    .map(|wp| wp.distance(center))
                    .sum::<f64>()
                    / 4.
                    * 2.
                    / 2f64.sqrt();
                DAffine2::from_scale_angle_translation(DVec2::splat(scale), 0., center)
            }
        }
    }
}

const OBJECT_TYPE_TABLE: TableDefinition<Uuid, &'static str> =
    TableDefinition::new("object_type_table_v1");
const OBJECT_DATA_TABLE: TableDefinition<Uuid, (&[u8], bool)> =
    TableDefinition::new("object_data_table_v1");
/// Layer group of grouped objects (absent = no group).
const OBJECT_GROUP_TABLE: TableDefinition<Uuid, &str> = TableDefinition::new("object_group_table_v1");

fn write_group(txn: &redb::WriteTransaction, uuid: Uuid, group: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let mut table = txn.open_table(OBJECT_GROUP_TABLE)?;
    let old = table.get(uuid)?.map(|v| v.value().to_string());
    if old.as_deref() != group {
        match group {
            Some(g) => {
                table.insert(uuid, g)?;
            }
            None => {
                table.remove(uuid)?;
            }
        }
    }
    Ok(())
}
impl Persistant for Object {
    fn db_update<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uuid = self.uuid();
        let mut data_table = txn.open_table(OBJECT_DATA_TABLE)?;
        let mut data = data_table
            .get_mut(uuid)?
            .ok_or("uuid in object_type_table_v1 did not exist")?;
        match self {
            Object::Gds {
                image,
                path,
                hidden,
            } => {
                let (db_name, db_hidden) = data.value();
                if db_name != path.as_os_str().as_encoded_bytes() || db_hidden != *hidden {
                    data.insert((path.as_os_str().as_encoded_bytes(), *hidden))?;
                }
                image.db_update(txn)?;
            }
            Object::File {
                image,
                path,
                hidden,
            } => {
                let (db_name, db_hidden) = data.value();
                if db_name != path.as_os_str().as_encoded_bytes() || db_hidden != *hidden {
                    data.insert((path.as_os_str().as_encoded_bytes(), *hidden))?;
                }
                image.db_update(txn)?;
            }
            Object::ScanImage {
                image,
                name,
                hidden,
                group,
            } => {
                let (db_name, db_hidden) = data.value();
                if db_name != name.as_bytes() || db_hidden != *hidden {
                    data.insert((name.as_bytes(), *hidden))?;
                }
                write_group(txn, uuid, group.as_deref())?;
                image.db_update(txn)?;
            }
            Object::ScanArea { image, hidden } => {
                let (_, db_hidden) = data.value();
                if db_hidden != *hidden {
                    data.insert((&[][..], *hidden))?;
                }
                image.db_update(txn)?;
            }
        }
        Ok(())
    }

    fn db_remove<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut type_table = txn.open_table(OBJECT_TYPE_TABLE)?;
        let mut data_table = txn.open_table(OBJECT_DATA_TABLE)?;
        {
            let type_name = type_table.get(id)?.ok_or("id should exist")?;
            match type_name.value() {
                "gds" => {
                    GDSImage::db_remove(id, txn)?;
                }
                "file" => {
                    FileImage::db_remove(id, txn)?;
                }
                "scanimage" => {
                    LiveImage::db_remove(id, txn)?;
                }
                "scanarea" => {
                    ScanArea::db_remove(id, txn)?;
                }
                invalid => return Err(format!("invalid type name {invalid}").into()),
            }
        }
        type_table.remove(id)?;
        data_table.remove(id)?;
        txn.open_table(OBJECT_GROUP_TABLE)?.remove(id)?;
        Ok(())
    }

    fn db_insert<'t>(
        &self,
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uuid = self.uuid();
        let mut data_table = txn.open_table(OBJECT_DATA_TABLE)?;
        let mut type_table = txn.open_table(OBJECT_TYPE_TABLE)?;
        match self {
            Object::Gds {
                image,
                path,
                hidden,
            } => {
                data_table.insert(uuid, (path.as_os_str().as_encoded_bytes(), *hidden))?;
                image.db_insert(txn)?;
                type_table.insert(uuid, "gds")?;
            }
            Object::File {
                image,
                path,
                hidden,
            } => {
                data_table.insert(uuid, (path.as_os_str().as_encoded_bytes(), *hidden))?;
                image.db_insert(txn)?;
                type_table.insert(uuid, "file")?;
            }
            Object::ScanImage {
                image,
                name,
                hidden,
                group,
            } => {
                data_table.insert(uuid, (name.as_bytes(), *hidden))?;
                write_group(txn, uuid, group.as_deref())?;
                image.db_insert(txn)?;
                type_table.insert(uuid, "scanimage")?;
            }
            Object::ScanArea { image, hidden } => {
                data_table.insert(uuid, (&[][..], *hidden))?;
                image.db_insert(txn)?;
                type_table.insert(uuid, "scanarea")?;
            }
        }
        Ok(())
    }
    fn db_read<'t>(
        id: Uuid,
        txn: &'t redb::WriteTransaction,
        encoder: &ImageEncoder,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let data_table = txn.open_table(OBJECT_DATA_TABLE)?;
        let type_table = txn.open_table(OBJECT_TYPE_TABLE)?;
        let type_name = type_table.get(id)?.ok_or("id should exist")?;
        let data = data_table.get(id)?.ok_or("id should exist")?;
        match type_name.value() {
            "gds" => {
                let image = GDSImage::db_read(id, txn, encoder)?;
                let (name, hidden) = data.value();
                let os_str = unsafe { OsStr::from_encoded_bytes_unchecked(name) };
                Ok(Self::Gds {
                    image,
                    path: PathBuf::from(os_str),
                    hidden,
                })
            }
            "file" => {
                let image = FileImage::db_read(id, txn, encoder)?;
                let (name, hidden) = data.value();
                let os_str = unsafe { OsStr::from_encoded_bytes_unchecked(name) };
                Ok(Self::File {
                    image,
                    path: PathBuf::from(os_str),
                    hidden,
                })
            }
            "scanimage" => {
                let image = LiveImage::db_read(id, txn, encoder)?;
                let (name, hidden) = data.value();
                let name = unsafe { str::from_utf8_unchecked(name) }.to_string();
                let group = txn.open_table(OBJECT_GROUP_TABLE)?.get(id)?.map(|g| g.value().to_string());
                Ok(Self::ScanImage {
                    image,
                    name,
                    hidden,
                    group,
                })
            }
            "scanarea" => {
                let image = ScanArea::db_read(id, txn, encoder)?;
                let (_, hidden) = data.value();
                Ok(Self::ScanArea { image, hidden })
            }
            invalid => return Err(format!("invalid type name {invalid}").into()),
        }
    }

    fn db_dump_stats<'t>(
        txn: &'t redb::WriteTransaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Object:");
        let type_len = txn.open_table(OBJECT_TYPE_TABLE)?.len()?;
        let data_len = txn.open_table(OBJECT_DATA_TABLE)?.len()?;
        println!("  type table: {type_len} items");
        println!("  data table: {data_len} items");
        GDSImage::db_dump_stats(txn)?;
        FileImage::db_dump_stats(txn)?;
        ScanArea::db_dump_stats(txn)?;
        LiveImage::db_dump_stats(txn)?;
        Ok(())
    }
}
