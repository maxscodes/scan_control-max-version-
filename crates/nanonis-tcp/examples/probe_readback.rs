use nanonis_tcp::blocking::NanonisTcp;
fn main() {
    let mut c = NanonisTcp::new(("localhost", 6500)).expect("connect");
    println!("bias     = {:?}", c.bias_get());
    println!("setpoint = {:?}", c.zctrl_setpnt_get());
    println!("current  = {:?}", c.current_get());
}
