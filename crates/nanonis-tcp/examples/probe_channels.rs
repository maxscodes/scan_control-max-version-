use nanonis_tcp::blocking::NanonisTcp;
fn main() {
    let mut c = NanonisTcp::new(("localhost", 6500)).expect("connect");
    let buf = c.scan_buffer_get().expect("buffer");
    println!("BufferGet indexes: {:?}", buf.channel_indexes);
    let names = c.signals_names_get().expect("names").names;
    for &i in &buf.channel_indexes {
        println!("  full[{i}] = {:?}", names.get(i as usize));
    }
    match c.signals_in_slots_get() {
        Ok(s) => {
            println!("InSlotsGet ({} names):", s.names.len());
            for (k, (n, i)) in s.names.iter().zip(&s.indexes).enumerate() {
                println!("  slot {k}: {n:?} (full idx {i})");
            }
        }
        Err(e) => println!("InSlotsGet error: {e:?}"),
    }
    for &i in &buf.channel_indexes {
        match c.scan_frame_data_grab(i as u32, nanonis_tcp::LineDir::Forward) {
            Ok(r) => println!("FrameDataGrab({i}) channel_name = {:?}", r.channel_name),
            Err(e) => println!("FrameDataGrab({i}) error: {e:?}"),
        }
    }
}
