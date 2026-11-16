use nanonis_tcp::{LineDir, ScanDir};


#[derive(Debug, Clone, Copy)]
pub struct ScanStatus {
    pub scan_dir: ScanDir,
    #[allow(dead_code)] // set by the end-of-line wait, not available on Nanonis v5
    pub line_dir: LineDir,
    #[allow(dead_code)] // set by the end-of-line wait, not available on Nanonis v5
    pub line_number: u32,
    pub scanning: bool,
}
impl Default for ScanStatus {
    fn default() -> Self {
        Self {
            scan_dir: ScanDir::Down,
            line_number: Default::default(),
            line_dir: LineDir::Forward,
            scanning: false,
        }
    }
}
impl ScanStatus {
    #[allow(dead_code)] // end-of-line wait: not available on Nanonis v5
    pub fn scan_line_position(&self, scan_size: [u32; 2], line_dir: LineDir) -> Option<f64> {
        if !self.scanning {
            return None;
        }
        let mut line_number = self.line_number;
        if line_dir == LineDir::Backward && self.line_dir == LineDir::Forward {
            line_number = line_number.saturating_sub(1)
        }
        let num_rows = scan_size[1];
        if line_number == num_rows {
            return None;
        }
        let mut pos = ((line_number as f64 - 0.5) / num_rows as f64) - 0.5;
        if self.scan_dir == ScanDir::Up {
            pos *= -1.;
        }
        pos = pos.clamp(-0.5,0.5);
        Some(pos)
    }
}
