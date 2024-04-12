mod banana;
mod nonlin2d;
mod nonlin3d;
mod nonlin4d;
mod qp;
mod quad3d;
mod quad4d;

#[cfg(test)]
#[ctor::ctor]
fn init() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        // .filter_level(log::LevelFilter::Trace)
        .format_module_path(false)
        .format_timestamp(None)
        .format_target(false)
        // .is_test(true)
        .init();
}
