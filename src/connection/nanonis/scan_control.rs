use egui::{Color32, Ui};
use nanonis_tcp::ScanDir;
use tracing::info;

use crate::connection::nanonis::{command_channel::CommandChannelSender, ScanStatus};

#[derive(Debug, Clone, Copy)]
pub enum ScanCommand {
    Start(ScanDir),
    Pause,
    Resume,
    Stop,
}

pub struct ScanControlState {
    sender: CommandChannelSender<ScanCommand, ()>,
    pending: bool,
    paused: bool,
}

impl ScanControlState {
    pub fn new(sender: &CommandChannelSender<ScanCommand, ()>) -> Self {
        Self {
            sender: sender.clone(),
            pending: false,
            paused: false,
        }
    }

    /// `sequence_active` only adds a marker; the buttons stay usable so a
    /// sequence scan can be paused/stopped/restarted by hand.
    pub fn show_menu(&mut self, ui: &mut Ui, scan_status: &ScanStatus, sequence_active: bool) {
        if self.sender.poll_complete().is_some() {
            self.pending = false;
        }
        // Scan.StatusGet has no paused flag (and may report a paused scan as
        // not running), so pause is tracked locally. It is only cleared by our
        // own Start/Resume/Stop commands. All buttons stay enabled so a pause
        // made in Nanonis can still be resumed or stopped from here.
        let mut command = None;
        ui.add_enabled_ui(!self.pending, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Scan Up ↑").clicked() {
                    command = Some(ScanCommand::Start(ScanDir::Up));
                }
                if ui.button("Scan Down ↓").clicked() {
                    command = Some(ScanCommand::Start(ScanDir::Down));
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Pause").clicked() {
                    command = Some(ScanCommand::Pause);
                }
                if ui.button("Resume").clicked() {
                    command = Some(ScanCommand::Resume);
                }
                if ui.button("Stop").clicked() {
                    command = Some(ScanCommand::Stop);
                }
            });
        });

        ui.horizontal(|ui| {
            let dir = match scan_status.scan_dir {
                ScanDir::Up => "↑",
                ScanDir::Down => "↓",
            };
            let status = if self.paused {
                format!("Paused {dir}")
            } else if scan_status.scanning {
                format!("Scanning {dir}")
            } else {
                "Idle".to_string()
            };
            ui.label(status);
            if sequence_active {
                ui.colored_label(Color32::ORANGE, "(sequence)");
            }
            if self.pending {
                ui.spinner();
            }
        });

        if let Some(cmd) = command {
            info!("Sending scan command {cmd:?}");
            self.pending = true;
            match cmd {
                ScanCommand::Pause => self.paused = true,
                ScanCommand::Resume | ScanCommand::Stop | ScanCommand::Start(_) => {
                    self.paused = false
                }
            }
            self.sender.send(cmd);
        }
    }
}
