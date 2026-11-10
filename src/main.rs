#![windows_subsystem = "windows"]

use std::error::Error;

use eframe::wgpu::{DeviceDescriptor, PresentMode};
use eyre::Context as _;
use tracing::{info, trace};
use tracing_subscriber::EnvFilter;
use wgpu::{
    Adapter, Backends, Device, FeaturesWGPU, FeaturesWebGPU, Instance, PowerPreference, Queue,
    RequestAdapterOptions,
};

mod analysis;
mod app;
mod components;
mod connection;
mod scan_view;
mod undo_queue;
mod utils;
mod view_object;
mod project;

fn main() -> Result<(), Box<dyn Error>> {
    attach_parent_console();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let (instance, adapter, device, queue) =
        init_wgpu().map_err(|e| eframe::Error::AppCreation(e.into()))?;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            present_mode: PresentMode::AutoNoVsync,
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::Existing(
                eframe::egui_wgpu::WgpuSetupExisting {
                    instance,
                    adapter,
                    device,
                    queue,
                },
            ),
            ..Default::default()
        },
        multisampling: 4,
        ..Default::default()
    };
    // let runtime = tokio::runtime::Runtime::new()?;
    // runtime.block_on(async {
    //     tokio::task::block_in_place(|| {
    //         eframe::run_native(
    //             "Scan Control",
    //             native_options,
    //             Box::new(|cc| {
    //                 egui_extras::install_image_loaders(&cc.egui_ctx);
    //                 cc.egui_ctx
    //                     .tessellation_options_mut(|opt| opt.feathering = false);
    //                 Ok(Box::new(app::MyApp::new(cc)))
    //             }),
    //         )
    //     })
    // })?;
    eframe::run_native(
        "Scan Control",
        native_options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            cc.egui_ctx
                .tessellation_options_mut(|opt| opt.feathering = false);
            Ok(Box::new(app::MyApp::new(cc)))
        }),
    )?;
    Ok(())
}

fn init_wgpu() -> eyre::Result<(Instance, Adapter, Device, Queue)> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(Backends::all());
    for adapter in adapters {
        trace!("Adapter: {}", adapter_to_str(&adapter));
    }
    let adapter = smol::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .context("Adapter request failed")?;
    info!("Selected: {}", adapter_to_str(&adapter));
    let (dev, queue) = smol::block_on(adapter.request_device(&DeviceDescriptor {
        required_features: wgpu::Features {
            features_wgpu: FeaturesWGPU::TIMESTAMP_QUERY_INSIDE_PASSES
                | FeaturesWGPU::SHADER_INT64
                | FeaturesWGPU::SHADER_F64,
            features_webgpu: FeaturesWebGPU::FLOAT32_FILTERABLE | FeaturesWebGPU::TIMESTAMP_QUERY,
        },
        ..Default::default()
    }))
    .context("Device request failed")?;
    Ok((instance, adapter, dev, queue))
}

fn adapter_to_str(adapter: &Adapter) -> String {
    let info = adapter.get_info();
    format!(
        "{} : {} : {} : {}",
        info.name,
        info.backend.to_str(),
        info.driver,
        info.driver_info
    )
}


#[cfg(windows)]
fn attach_parent_console() {
    use windows::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};

    unsafe {
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

#[cfg(not(windows))]
fn attach_parent_console() {}