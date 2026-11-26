use std::path::Path;

use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = ort::init_from(Path::new(
        r"D:\bgiprro\bettergi-rs\vendor\ort\onnxruntime.dll",
    ))?
    .with_name("inspect-mining")
    .build()?;
    let session = Session::builder(&env)?.commit_from_file(
        r"D:\bgiprro\bettergi-libraries\BetterGI.Assets.Model\Assets\Model\Mine\bgi_mine.onnx",
    )?;
    for input in session.inputs() {
        println!("input {:?}", input);
    }
    for output in session.outputs() {
        println!("output {:?}", output);
    }
    Ok(())
}
