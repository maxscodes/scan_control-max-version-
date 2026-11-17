use std::{
    ops::RangeBounds,
    sync::{Arc, OnceLock},
};

use glam::DAffine2;
use itertools::{Itertools, izip};
pub use shaders::plane_fit::NormalizeData;
use tracing::info;
use wgpu::{
    BufferUsages, CommandEncoder, ComputePass, ComputePipeline, Device, Extent3d, QuerySet,
    QueryType, Queue, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
    util::DeviceExt,
    wgt::{QuerySetDescriptor, TextureDataOrder},
};

use crate::{
    buffers::{BufferOpError, StorageBuffer, TransformBuffer, Unalign},
    shaders::{self, scan_image::NormalizeControl},
};

#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    FullScale,
    StdDev(f32),
}
impl From<NormalizationType> for NormalizeControl {
    fn from(value: NormalizationType) -> Self {
        NormalizeControl {
            max_min: matches!(value, NormalizationType::FullScale) as u32,
            _pad: 0,
            std_dev_mul: match value {
                NormalizationType::FullScale => 0.,
                NormalizationType::StdDev(f) => f,
            },
            opacity: 1.,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WriteLinesError {
    #[error("requested to write `{requested}` lines, but there is only room for `{remaining}`")]
    TooManyLines { requested: usize, remaining: usize },
    #[error("{0}")]
    BufferOpError(#[from] BufferOpError),
}

#[derive(Clone)]
pub struct ImageComputeBuffers {
    size: [u32; 2],
    world_transform_buffer: TransformBuffer,
    image_size_buffer: StorageBuffer<u32>,
    image_data_buffer: StorageBuffer<f32>,
    planarize_buffer: StorageBuffer<f32>,
    normalize_buffer: StorageBuffer<shaders::plane_fit::NormalizeData>,
    normalize_control_buffer: StorageBuffer<shaders::scan_image::NormalizeControl>,
    image_src_bg: Arc<shaders::plane_fit::bind_groups::BindGroup0>,
    normalize_bg: Arc<shaders::plane_fit::bind_groups::BindGroup1>,
    pub(crate) scan_image_bg: Arc<shaders::scan_image::bind_groups::BindGroup1>,
}
impl ImageComputeBuffers {
    pub fn new(device: &Device, queue: &Queue, label: Option<&str>, size: [u32; 2]) -> Self {
        let size_buffer_label = label.map(|name| format!("{name}_size_buffer"));
        let image_size_buffer = StorageBuffer::new(
            device,
            size_buffer_label.as_deref(),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size.as_slice(),
        );
        let data_buffer_label = label.map(|name| format!("{name}_data_buffer"));
        let image_data_buffer = StorageBuffer::new_uninit(
            device,
            data_buffer_label.as_deref(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            size[0] as usize * size[1] as usize,
        );
        let world_transform_buffer = TransformBuffer::new(device);
        let image_texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[TextureFormat::R32Float],
            },
            TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(vec![f32::NAN; size[0] as usize * size[1] as usize].as_slice()),
        );
        let normalize_control_buffer = StorageBuffer::new(
            device,
            None,
            BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            &[NormalizationType::StdDev(3.).into()],
        );
        let normalize_buffer = StorageBuffer::<shaders::plane_fit::NormalizeData>::new_uninit(
            device,
            Some("normalize_out"),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::UNIFORM,
            1,
        );
        let planarize_buffer = StorageBuffer::<f32>::new_uninit(
            device,
            Some("planarize_out"),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            size[0] as usize * size[1] as usize,
        );
        let image_src_bg = Arc::new(shaders::plane_fit::bind_groups::BindGroup0::from_bindings(
            device,
            shaders::plane_fit::bind_groups::BindGroupLayout0 {
                image_size: image_size_buffer.as_entire_buffer_binding(),
                image_in: image_data_buffer.as_entire_buffer_binding(),
            },
        ));
        let scan_image_bg = Arc::new(shaders::scan_image::bind_groups::BindGroup1::from_bindings(
            device,
            shaders::scan_image::bind_groups::BindGroupLayout1 {
                quad2world: world_transform_buffer.as_entire_buffer_binding(),
                height_map: &image_texture.create_view(&TextureViewDescriptor::default()),
                normalize_data: normalize_buffer.as_entire_buffer_binding(),
                normalize_control: normalize_control_buffer.as_entire_buffer_binding(),
            },
        ));
        let normalize_bg = Arc::new(shaders::plane_fit::bind_groups::BindGroup1::from_bindings(
            device,
            shaders::plane_fit::bind_groups::BindGroupLayout1 {
                texture_out: &image_texture.create_view(&TextureViewDescriptor::default()),
                planarize_out: planarize_buffer.as_entire_buffer_binding(),
                normalize_out: normalize_buffer.as_entire_buffer_binding(),
            },
        ));
        Self {
            size,
            image_size_buffer,
            image_data_buffer,
            world_transform_buffer,
            planarize_buffer,
            normalize_buffer,
            normalize_control_buffer,
            scan_image_bg,
            image_src_bg,
            normalize_bg,
        }
    }
    pub fn write_world_transform(&self, queue: &Queue, transform: DAffine2) {
        self.world_transform_buffer.queue_write(queue, transform);
    }
    pub fn write_normalization_type(
        &self,
        queue: &Queue,
        normalization_type: NormalizationType,
    ) -> Result<(), BufferOpError> {
        let mut buf = self
            .normalize_control_buffer
            .queue_write_with(queue, 0, 1)?;
        buf[0] = normalization_type.into();
        Ok(())
    }
    /// Contrast and opacity (0..1) used when drawing the image.
    pub fn write_display(
        &self,
        queue: &Queue,
        normalization_type: NormalizationType,
        opacity: f32,
    ) -> Result<(), BufferOpError> {
        let mut buf = self
            .normalize_control_buffer
            .queue_write_with(queue, 0, 1)?;
        let mut c: NormalizeControl = normalization_type.into();
        c.opacity = opacity.clamp(0., 1.);
        buf[0] = c;
        Ok(())
    }
    pub fn write_lines_range(
        &self,
        queue: &Queue,
        lines: impl RangeBounds<u32>,
        callback: impl Fn(&mut [f32]),
    ) -> Result<(), BufferOpError> {
        let offset = match lines.start_bound() {
            std::ops::Bound::Included(v) => *v,
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(_) => panic!("don't use excluded start bound for lines"),
        };
        let size = match lines.end_bound() {
            std::ops::Bound::Included(v) => v - offset,
            std::ops::Bound::Excluded(v) => v + 1 - offset,
            std::ops::Bound::Unbounded => self.size[1] - offset,
        };
        let mut buf = self.image_data_buffer.queue_write_with(
            queue,
            (offset * self.size[0]) as usize,
            (size * self.size[0]) as usize,
        )?;
        callback(buf.as_mut());
        Ok(())
    }
    pub fn size(&self) -> [u32; 2] {
        self.size
    }
    pub fn download_normalize_data(
        &self,
        device: &Device,
        queue: &Queue,
        callback: impl FnOnce(NormalizeData) + Send + 'static,
    ) -> Result<(), BufferOpError> {
        self.normalize_buffer
            .queue_download(device, queue, ..1, |data| callback(data[0]))
    }
    pub fn download_fit_data(
        &self,
        device: &Device,
        queue: &Queue,
        fit_type: FitType,
        callback: impl FnOnce(FitData) + Send + 'static,
    ) -> Result<(), BufferOpError> {
        let size = self.size();
        self.planarize_buffer.queue_download(
            device,
            queue,
            ..fit_type.download_size(size),
            move |data| callback(FitData::from_raw(data, size, fit_type)),
        )
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive, num_enum::IntoPrimitive,
)]
#[repr(u8)]
pub enum FitType {
    MeanSubtract = 0,
    PlaneFitSubtract = 1,
    LineMeanSubtract = 2,
    LineFitSubtract = 3,
}
impl FitType {
    fn download_size(&self, size: [u32; 2]) -> usize {
        match self {
            FitType::MeanSubtract => 2,
            FitType::PlaneFitSubtract => 4,
            FitType::LineMeanSubtract => size[1] as usize * 2,
            FitType::LineFitSubtract => size[1] as usize * 3,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FitData {
    MeanSubtract {
        mean: f32,
    },
    PlaneFitSubtract {
        mean: f32,
        x_slope: f32,
        y_slope: f32,
    },
    LineMeanSubtract {
        means: Box<[f32]>,
    },
    LineFitSubtract {
        means: Box<[f32]>,
        slopes: Box<[f32]>,
    },
}
impl FitData {
    pub fn mean(&self) -> f32 {
        match self {
            FitData::MeanSubtract { mean } => *mean,
            FitData::PlaneFitSubtract { mean, .. } => *mean,
            FitData::LineMeanSubtract { means } => {
                let (sum, count) = means
                    .iter()
                    .filter(|m| !m.is_nan())
                    .fold((0., 0.), |(sum, count), v| (sum + v, count + 1.));
                sum / count
            }
            FitData::LineFitSubtract { means, .. } => {
                let (sum, count) = means
                    .iter()
                    .filter(|m| !m.is_nan())
                    .fold((0., 0.), |(sum, count), v| (sum + v, count + 1.));
                sum / count
            }
        }
    }
    fn from_raw(data: &[f32], size: [u32; 2], fit_type: FitType) -> Self {
        assert_eq!(data.len(), fit_type.download_size(size));
        let h = size[1] as usize;
        match fit_type {
            FitType::MeanSubtract => Self::MeanSubtract {
                mean: data[0] / data[1],
            },
            FitType::PlaneFitSubtract => Self::PlaneFitSubtract {
                mean: data[0] / data[1],
                x_slope: data[2],
                y_slope: data[3],
            },
            FitType::LineMeanSubtract => {
                let sums = &data[0 * h..][..h];
                let counts = &data[1 * h..][..h];
                Self::LineMeanSubtract {
                    means: izip!(sums, counts)
                        .map(|(sum, count)| sum / count)
                        .collect_vec()
                        .into_boxed_slice(),
                }
            }
            FitType::LineFitSubtract => {
                let sums = &data[0 * h..][..h];
                let counts = &data[1 * h..][..h];
                let slopes = &data[2 * h..][..h];
                Self::LineFitSubtract {
                    means: izip!(sums, counts)
                        .map(|(sum, count)| sum / count)
                        .collect_vec()
                        .into_boxed_slice(),
                    slopes: slopes.to_vec().into_boxed_slice(),
                }
            }
        }
    }
}

#[derive(Clone)]
struct ScratchBuffers {
    size: [u32; 2],
    x_sum: StorageBuffer<f64>,
    y_sum: StorageBuffer<f64>,
    count_buf: StorageBuffer<u32>,
    mins: StorageBuffer<f64>,
    maxs: StorageBuffer<f64>,
    std_devs: StorageBuffer<f64>,
    xz: StorageBuffer<f64>,
    yz: StorageBuffer<f64>,
    xx: StorageBuffer<f64>,
    yy: StorageBuffer<f64>,
    bg_2: Arc<shaders::plane_fit::bind_groups::BindGroup2>,
    bg_3: Arc<shaders::plane_fit::bind_groups::BindGroup3>,
}
impl ScratchBuffers {
    fn new(device: &Device, size: [u32; 2]) -> Self {
        let count_buf = StorageBuffer::new_uninit(
            device,
            Some("count_buf"),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            size[0] as usize * size[1] as usize,
        );
        let mk_buffer = |s: &'static str| {
            StorageBuffer::new_uninit(
                device,
                Some(s),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                size[0] as usize * size[1] as usize,
            )
        };
        let x_sum = mk_buffer("x_sum");
        let y_sum = mk_buffer("y_sum");
        let mins = mk_buffer("mins");
        let maxs = mk_buffer("maxs");
        let std_devs = mk_buffer("std_devs");
        let xz = mk_buffer("xz");
        let yz = mk_buffer("yz");
        let xx = mk_buffer("xx");
        let yy = mk_buffer("yy");
        Self {
            size,
            bg_2: Arc::new(shaders::plane_fit::bind_groups::BindGroup2::from_bindings(
                device,
                shaders::plane_fit::bind_groups::BindGroupLayout2 {
                    x_sum: x_sum.as_entire_buffer_binding(),
                    y_sum: y_sum.as_entire_buffer_binding(),
                    count: count_buf.as_entire_buffer_binding(),
                    mins: mins.as_entire_buffer_binding(),
                    maxs: maxs.as_entire_buffer_binding(),
                    std_devs: std_devs.as_entire_buffer_binding(),
                },
            )),
            bg_3: Arc::new(shaders::plane_fit::bind_groups::BindGroup3::from_bindings(
                device,
                shaders::plane_fit::bind_groups::BindGroupLayout3 {
                    xz: xz.as_entire_buffer_binding(),
                    yz: yz.as_entire_buffer_binding(),
                    xx: xx.as_entire_buffer_binding(),
                    yy: yy.as_entire_buffer_binding(),
                },
            )),
            x_sum,
            y_sum,
            xz,
            yz,
            xx,
            yy,
            count_buf,
            mins,
            maxs,
            std_devs,
        }
    }
}

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct ImageComputePipeline {
    copy_image: ComputePipeline,
    copy_image_transpose: ComputePipeline,
    generate_sums_plane: ComputePipeline,
    generate_sums_lines: ComputePipeline,
    reduce_image: ComputePipeline,
    reduce_image_lines: ComputePipeline,
    reduce_sums_plane: ComputePipeline,
    reduce_sums_lines: ComputePipeline,
    reduce_normalizations: ComputePipeline,
    generate_normalization__mean_subtract: ComputePipeline,
    generate_normalization__plane_fit: ComputePipeline,
    generate_normalization__line_fit: ComputePipeline,
    generate_normalization__line_mean: ComputePipeline,
    clear_texture: ComputePipeline,
    qs: QuerySet,
    qs_buf: StorageBuffer<Unalign<u64>>,
    scratch_buffers: Arc<parking_lot::Mutex<ScratchBuffers>>,
}
impl ImageComputePipeline {
    pub fn new(device: &Device) -> Self {
        let n_timings = 8;
        Self {
            copy_image: shaders::plane_fit::compute::create_copy_image_pipeline(device),
            copy_image_transpose: shaders::plane_fit::compute::create_copy_image_transpose_pipeline(
                device,
            ),
            generate_sums_plane: shaders::plane_fit::compute::create_generate_sums_plane_pipeline(
                device,
            ),
            generate_sums_lines: shaders::plane_fit::compute::create_generate_sums_lines_pipeline(
                device,
            ),
            reduce_image: shaders::plane_fit::compute::create_reduce_image_pipeline(device),
            reduce_image_lines: shaders::plane_fit::compute::create_reduce_image_lines_pipeline(
                device,
            ),
            reduce_sums_plane: shaders::plane_fit::compute::create_reduce_sums_plane_pipeline(
                device,
            ),
            reduce_sums_lines: shaders::plane_fit::compute::create_reduce_sums_lines_pipeline(
                device,
            ),
            reduce_normalizations:
                shaders::plane_fit::compute::create_reduce_normalizations_pipeline(device),
            generate_normalization__mean_subtract:
                shaders::plane_fit::compute::create_generate_normalization__mean_subtract_pipeline(
                    device,
                ),
            generate_normalization__plane_fit:
                shaders::plane_fit::compute::create_generate_normalization__plane_fit_pipeline(
                    device,
                ),
            generate_normalization__line_fit:
                shaders::plane_fit::compute::create_generate_normalization__line_fit_pipeline(
                    device,
                ),
            generate_normalization__line_mean:
                shaders::plane_fit::compute::create_generate_normalization__line_mean_pipeline(
                    device,
                ),
            clear_texture: shaders::plane_fit::compute::create_clear_texture_pipeline(device),
            qs: device.create_query_set(&QuerySetDescriptor {
                label: Some("plane_fitter_qs"),
                ty: QueryType::Timestamp,
                count: n_timings * 2,
            }),
            qs_buf: StorageBuffer::new_uninit(
                device,
                Some("plane_fitter_qs_buf"),
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                n_timings as usize * 2,
            ),
            scratch_buffers: Arc::new(parking_lot::Mutex::new(ScratchBuffers::new(
                device,
                [1024, 1024],
            ))),
        }
    }
    pub fn dispatch(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
        fit_type: FitType,
    ) {
        let dispatch_fn = match fit_type {
            FitType::MeanSubtract => Self::dispatch_mean_subtract,
            FitType::PlaneFitSubtract => Self::dispatch_plane_fit_subtract,
            FitType::LineMeanSubtract => Self::dispatch_line_mean_subtract,
            FitType::LineFitSubtract => Self::dispatch_line_fit_subtract,
        };
        dispatch_fn(self, device, pass, image_buffers);
    }
    pub fn dispatch_mean_subtract(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
    ) -> usize {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n as u32);
            qs_n += 1;
        };
        {
            let mut scratch = self.scratch_buffers.lock();
            if izip!(scratch.size, image_buffers.size).any(|(a, b)| a < b) {
                info!("Reallocating scratch buffers to {:?}", image_buffers.size);
                *scratch = ScratchBuffers::new(device, image_buffers.size);
            }
            scratch.bg_2.set(pass);
            scratch.bg_3.set(pass);
        }
        image_buffers.image_src_bg.set(pass);
        image_buffers.normalize_bg.set(pass);

        let size = image_buffers.size();
        pass.push_debug_group("mean subtract");

        pass.set_pipeline(&self.copy_image);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_image);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_normalization__mean_subtract);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_normalizations);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.pop_debug_group();
        qs_n / 2
    }
    pub fn dispatch_plane_fit_subtract(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
    ) -> usize {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n as u32);
            qs_n += 1;
        };
        {
            let mut scratch = self.scratch_buffers.lock();
            if izip!(scratch.size, image_buffers.size).any(|(a, b)| a < b) {
                info!("Reallocating scratch buffers to {:?}", image_buffers.size);
                *scratch = ScratchBuffers::new(device, image_buffers.size);
            }
            scratch.bg_2.set(pass);
            scratch.bg_3.set(pass);
        }
        image_buffers.image_src_bg.set(pass);
        image_buffers.normalize_bg.set(pass);

        pass.push_debug_group("plane fit");
        let size = image_buffers.size();

        pass.set_pipeline(&self.copy_image);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_image);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_sums_plane);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_sums_plane);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_normalization__plane_fit);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_normalizations);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.pop_debug_group();
        qs_n / 2
    }
    pub fn dispatch_line_fit_subtract(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
    ) -> usize {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n as u32);
            qs_n += 1;
        };
        {
            let mut scratch = self.scratch_buffers.lock();
            if izip!(scratch.size, image_buffers.size).any(|(a, b)| a < b) {
                info!("Reallocating scratch buffers to {:?}", image_buffers.size);
                *scratch = ScratchBuffers::new(device, image_buffers.size);
            }
            scratch.bg_2.set(pass);
            scratch.bg_3.set(pass);
        }
        image_buffers.image_src_bg.set(pass);
        image_buffers.normalize_bg.set(pass);

        pass.push_debug_group("line fit subtract");
        let size = image_buffers.size();

        pass.set_pipeline(&self.copy_image_transpose);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_image_lines);
        wts(pass);
        dispatch_y_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_sums_lines);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_sums_lines);
        wts(pass);
        dispatch_y_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_normalization__line_fit);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_normalizations);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);

        pass.pop_debug_group();
        qs_n / 2
    }
    pub fn dispatch_line_mean_subtract(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
    ) -> usize {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n as u32);
            qs_n += 1;
        };
        {
            let mut scratch = self.scratch_buffers.lock();
            if izip!(scratch.size, image_buffers.size).any(|(a, b)| a < b) {
                info!("Reallocating scratch buffers to {:?}", image_buffers.size);
                *scratch = ScratchBuffers::new(device, image_buffers.size);
            }
            scratch.bg_2.set(pass);
            scratch.bg_3.set(pass);
        }
        image_buffers.image_src_bg.set(pass);
        image_buffers.normalize_bg.set(pass);

        let size = image_buffers.size();
        pass.push_debug_group("line mean subtract");

        pass.set_pipeline(&self.copy_image_transpose);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_image_lines);
        wts(pass);
        dispatch_y_reduction(pass, size);
        wts(pass);

        pass.set_pipeline(&self.generate_normalization__line_mean);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        pass.set_pipeline(&self.reduce_normalizations);
        wts(pass);
        dispatch_reduction(pass, size);
        wts(pass);
        pass.pop_debug_group();

        qs_n / 2
    }
    pub fn dispatch_clear_texture(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        image_buffers: &ImageComputeBuffers,
    ) -> usize {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n as u32);
            qs_n += 1;
        };
        {
            let mut scratch = self.scratch_buffers.lock();
            if izip!(scratch.size, image_buffers.size).any(|(a, b)| a < b) {
                info!("Reallocating scratch buffers to {:?}", image_buffers.size);
                *scratch = ScratchBuffers::new(device, image_buffers.size);
            }
            scratch.bg_2.set(pass);
            scratch.bg_3.set(pass);
        }
        image_buffers.image_src_bg.set(pass);
        image_buffers.normalize_bg.set(pass);
        let size = image_buffers.size;

        pass.set_pipeline(&self.clear_texture);
        wts(pass);
        dispatch_linear(pass, size);
        wts(pass);

        qs_n / 2
    }
    pub fn queue_timings_download(
        &self,
        device: &Device,
        queue: &Queue,
        num: usize,
    ) -> Result<Arc<OnceLock<Box<[u64]>>>, BufferOpError> {
        let buf = Arc::new(OnceLock::new());
        let buf_clone = buf.clone();
        self.qs_buf
            .queue_download(device, queue, ..num * 2, move |data| {
                buf.set(
                    data.iter()
                        .map(Unalign::get)
                        .chunks(2)
                        .into_iter()
                        .map(|c| c.collect_tuple().unwrap())
                        .map(|(a, b)| b.saturating_sub(a))
                        .collect_vec()
                        .into_boxed_slice(),
                )
                .unwrap();
            })?;
        Ok(buf_clone)
    }
    pub fn resolve_timings(&self, encoder: &mut CommandEncoder, num: usize) {
        encoder.resolve_query_set(&self.qs, 0..num as u32 * 2, self.qs_buf.buffer_ref(), 0);
    }
}

fn dispatch_linear(pass: &mut ComputePass, size: [u32; 2]) {
    pass.dispatch_workgroups(
        num_workgroups(size[0] * size[1], shaders::plane_fit::WGS),
        1,
        1,
    );
}

fn dispatch_reduction(pass: &mut ComputePass, size: [u32; 2]) {
    let mut remaining_data = size[0] * size[1];
    while remaining_data > 1 {
        let num_wgs = num_workgroups(remaining_data, shaders::plane_fit::WGS);
        pass.dispatch_workgroups(num_wgs, 1, 1);
        remaining_data = num_wgs;
    }
}

fn dispatch_y_reduction(pass: &mut ComputePass, size: [u32; 2]) {
    let mut remaining_data = size[0];
    let num_wgs_cols = num_workgroups(size[1], shaders::plane_fit::WGS_SQUARE);
    while remaining_data > 1 {
        let num_wgs_rows = num_workgroups(remaining_data, shaders::plane_fit::WGS_SQUARE);
        pass.dispatch_workgroups(num_wgs_cols, num_wgs_rows, 1);
        remaining_data = num_wgs_rows;
    }
}

#[inline]
fn num_workgroups(num: u32, wg_size: u32) -> u32 {
    num.saturating_sub(1) / wg_size + 1
}

fn map_range<T, O>(range: impl RangeBounds<T>, f: impl Fn(&T) -> O) -> impl RangeBounds<O> {
    (range.start_bound().map(&f), range.end_bound().map(&f))
}
