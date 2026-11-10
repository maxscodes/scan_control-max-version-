use std::{
    io::{BufRead, BufReader, Read as _},
    path::{Path, PathBuf},
    str::Utf8Error,
    vec,
};

use eyre::{Context, ContextCompat, Result, bail, ensure};
use itertools::Itertools;

pub struct SXM {
    pub metadata: Vec<(Box<str>, Box<str>)>,
    pub data: Vec<[Box<[f32]>; 2]>,
    path: PathBuf,
}

impl SXM {
    pub fn get_metadata<'a>(&'a self, key: &str) -> Result<&'a str> {
        let Ok(i) = self.metadata.binary_search_by_key(&key, |(k, _)| k) else {
            bail!("file does not contain key `{key}`");
        };
        Ok(&self.metadata[i].1)
    }
    pub fn get_image_size(&self) -> Result<[u32; 2]> {
        let mut dims = self
            .get_metadata("SCAN_PIXELS")?
            .split_ascii_whitespace()
            .map(str::parse);
        Ok([
            dims.next()
                .wrap_err("SCAN_PIXELS x size missing")?
                .context("SCAN_PIXELS x size parse error")?,
            dims.next()
                .wrap_err("SCAN_PIXELS y size missing")?
                .context("SCAN_PIXELS y size parse error")?,
        ])
    }
    pub fn get_channels(&self) -> Result<Vec<(&str, ChannelInfo)>> {
        let mut lines = self.get_metadata("DATA_INFO")?.lines();
        let header = lines
            .next()
            .wrap_err("`DATA_INFO` is empty")?
            .trim()
            .split("\t")
            .collect_vec();
        let expected_header = [
            "Channel",
            "Name",
            "Unit",
            "Direction",
            "Calibration",
            "Offset",
        ];
        ensure!(
            header == expected_header,
            "expected `DATA_INFO` header to be `{expected_header:?}` but got `{header:?}`"
        );
        let mut out = vec![];
        for (i, line) in lines.enumerate() {
            let [channel, name, unit, dir, cal, off] = line
                .trim()
                .split("\t")
                .collect_array()
                .wrap_err_with(|| format!("wrong number of entries in row {i}"))?;
            ensure!(
                dir == "both",
                "expected `Direction` to be `both` but got `{dir}`"
            );
            out.push((
                name,
                ChannelInfo {
                    channel: channel
                        .parse()
                        .with_context(|| format!("invalid `Channel` value in row `{i}`"))?,
                    unit: unit.to_string(),
                    calibration: cal
                        .parse()
                        .with_context(|| format!("invalid `Calibration` value in row `{i}`"))?,
                    offset: off
                        .parse()
                        .with_context(|| format!("invalid `Offset` value in row `{i}`"))?,
                },
            ));
        }
        Ok(out)
    }
    pub fn get_scan_range(&self) -> Result<[f32; 2]> {
        let mut ranges = self
            .get_metadata("SCAN_RANGE")?
            .split_ascii_whitespace()
            .map(str::parse);
        Ok([
            ranges
                .next()
                .wrap_err("SCAN_RANGE x value missing")?
                .context("SCAN_RANGE x value parse error")?,
            ranges
                .next()
                .wrap_err("SCAN_RANGE y value missing")?
                .context("SCAN_RANGE y value parse error")?,
        ])
    }
    pub fn get_scan_center(&self) -> Result<[f32; 2]> {
        let mut pos = self
            .get_metadata("SCAN_OFFSET")?
            .split_ascii_whitespace()
            .map(str::parse);
        Ok([
            pos.next()
                .wrap_err("SCAN_OFFSET x value missing")?
                .context("SCAN_OFFSET x value parse error")?,
            pos.next()
                .wrap_err("SCAN_OFFSET y value missing")?
                .context("SCAN_OFFSET y value parse error")?,
        ])
    }
    pub fn get_scan_angle(&self) -> Result<f32> {
        let mut pos = self
            .get_metadata("SCAN_ANGLE")?
            .split_ascii_whitespace()
            .map(str::parse);
        pos.next()
            .wrap_err("SCAN_ANGLE value missing")?
            .context("SCAN_ANGLE value parse error")
    }
    pub fn get_datetime(&self) -> Result<chrono::NaiveDateTime> {
        let date_raw = self.get_metadata("REC_DATE")?;
        let date = chrono::NaiveDate::parse_from_str(date_raw, "%d.%m.%Y")
            .context("failed to parse date")?;
        let time_raw = self.get_metadata("REC_TIME")?;
        let time = chrono::NaiveTime::parse_from_str(time_raw, "%H:%M:%S")
            .context("failed to parse time")?;
        Ok(date.and_time(time))
    }
    pub fn get_name(&self) -> Result<&str> {
        self.path
            .file_stem()
            .context("path was not a file")?
            .to_str()
            .context("file name was not valid utf-8")
            .context("failed to get name from file")
    }
    pub fn rename(&mut self, new_name: &str) -> Result<()> {
        let new_path = self
            .path
            .parent()
            .unwrap()
            .to_path_buf()
            .join(format!("{new_name}.sxm"));
        if new_path.exists() {
            bail!("file `{}` already exists", new_path.to_str().unwrap())
        }
        std::fs::rename(&self.path, &new_path).context("failed to rename file")?;
        self.path = new_path;
        Ok(())
    }
    #[inline]
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = BufReader::new(std::fs::File::open(&path)?);

        let mut meta = Vec::new();
        reader.read_until(0x04, &mut meta)?;

        let mut out = Self {
            metadata: Self::parse_metadata(&meta).context("failed to parse metadata")?,
            data: vec![],
            path: path.as_ref().to_path_buf(),
        };

        let scanit_type = out.get_metadata("SCANIT_TYPE")?.replace(" ", "");
        ensure!(
            &scanit_type == "FLOATMSBFIRST",
            "expected SCANIT_TYPE to be `FLOATMSBFIRST` but got `{scanit_type}`"
        );

        let [pix_x, pix_y] = out.get_image_size()?;
        let num_channels = out
            .get_channels()
            .context("failed to parse channel info")?
            .len();
        let mut read_buf = [0u8; 4];
        for _ in 0..num_channels {
            let mut data_buf_forward = Vec::with_capacity((pix_x * pix_y) as usize);
            let mut data_buf_backward = Vec::with_capacity((pix_x * pix_y) as usize);
            for _ in 0..pix_x * pix_y {
                reader
                    .read_exact(&mut read_buf)
                    .context("got less data than expected")?;
                data_buf_forward.push(f32::from_be_bytes(read_buf));
            }
            for _ in 0..pix_x * pix_y {
                reader
                    .read_exact(&mut read_buf)
                    .context("got less data than expected")?;
                data_buf_backward.push(f32::from_be_bytes(read_buf));
            }
            out.data.push([
                data_buf_forward.into_boxed_slice(),
                data_buf_backward.into_boxed_slice(),
            ]);
        }
        let mut rest = vec![];
        reader.read_to_end(&mut rest)?;
        if !rest.is_empty() {
            bail!("got more data than expected");
        }
        Ok(out)
    }
    fn parse_metadata(mut input: &[u8]) -> Result<Vec<(Box<str>, Box<str>)>> {
        let mut out = vec![];
        let mut key = None::<&str>;
        let mut value = String::new();
        while let Some(line) = chop_line(&mut input) {
            let line = line.context("invalid metadata line found")?;
            if line.starts_with(':') && line.ends_with(':') {
                if let Some(key) = key {
                    out.push((key.into(), value[..value.len() - 1].into()));
                }
                key = Some(line.trim_matches(':'));
                value.clear();
            } else {
                value.push_str(line);
                value.push('\n');
            }
        }
        out.sort();
        Ok(out)
    }
}

#[derive(Debug)]
pub struct ChannelInfo {
    channel: u32,
    unit: String,
    calibration: f32,
    offset: f32,
}

fn chop_line<'a>(input: &mut &'a [u8]) -> Option<Result<&'a str, Utf8Error>> {
    if input.is_empty() {
        return None;
    }
    let (line, rest) = match input.iter().position(|c| *c == b'\n') {
        Some(end) => (&input[..end], &input[end + 1..]),
        None => (&input[..], &[][..]),
    };
    *input = rest;
    Some(std::str::from_utf8(line))
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        eyre::set_hook(Box::new(eyre::DefaultHandler::default_with));
        let sxm = SXM::parse_file("20240229_066.sxm").context("SXM file load failed")?;
        dbg!(sxm.data.len());
        dbg!(sxm.get_channels()?);
        Ok(())
    }

    #[test]
    fn test_chop_line() -> Result<()> {
        let mut s = r#" 
"#
        .as_bytes();
        while let Some(line) = chop_line(&mut s) {
            let line = line?;
            println!("line: \"{line}\"");
        }
        println!("rest: {s:?}");
        Ok(())
    }
}
