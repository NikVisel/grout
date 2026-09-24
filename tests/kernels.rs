use anyhow::Result;
use cuda_async::device_operation::{DeviceOp, value};
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::kernels::add_2d_f16;
use grout::kernels::kv_cache_update_seq_f16;
use std::sync::Arc;

#[test]
fn add_2d_kernel_compiles_and_executes() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("skipping CUDA kernel integration test: no CUDA devices found");
            return Ok(());
        }
        Err(err) => {
            eprintln!("skipping CUDA kernel integration test: CUDA unavailable: {err:?}");
            return Ok(());
        }
    }

    const BLOCK: usize = 4;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let lhs_host = Arc::new(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ]);
    let rhs_host = Arc::new(vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ]);

    let lhs = Arc::new(
        api::copy_host_vec_to_device(&lhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let rhs = Arc::new(
        api::copy_host_vec_to_device(&rhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let out = api::zeros::<f16>(&[1, BLOCK]).sync_on(&stream)?;

    let result = add_2d_f16(value(out.partition([1, BLOCK])), value(lhs), value(rhs))
        .generics(vec![BLOCK.to_string()])
        .sync_on(&stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(&stream)?;

    let actual: Vec<f32> = actual.into_iter().map(|x| x.to_f32()).collect();
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

#[test]
fn mtp_kv_suffix_write_preserves_prefix_and_unused_tail() -> Result<()> {
    if !matches!(Device::device_count(), Ok(n) if n > 0) {
        eprintln!("skipping CUDA suffix-write test: CUDA unavailable");
        return Ok(());
    }
    let stream = Device::new(0)?.new_stream()?;
    const HEADS: usize = 2;
    const CAPACITY: usize = 16;
    const D: usize = 4;
    const ROWS: usize = 5;
    const START: usize = 3;
    const BM: usize = 4;
    let values: Vec<_> = (0..ROWS * HEADS * D)
        .map(|i| f16::from_f32((i + 1) as f32))
        .collect();
    let incoming = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(values.clone()))
            .reshape(&[ROWS, HEADS, D])
            .sync_on(&stream)?,
    );
    let k = api::zeros::<f16>(&[HEADS, CAPACITY, D]).sync_on(&stream)?;
    let v = api::zeros::<f16>(&[HEADS, CAPACITY, D]).sync_on(&stream)?;
    let result = unsafe {
        kv_cache_update_seq_f16(
            value(incoming.clone()),
            value(incoming),
            value(k.partition([1, BM, D])),
            value(v.partition([1, BM, D])),
            value(START as i32),
            value(ROWS as i32),
        )
        .generics(vec![D.to_string(), D.to_string(), BM.to_string()])
        .sync_on(&stream)?
    };
    let k = result.2.unpartition().to_host_vec().sync_on(&stream)?;
    let v = result.3.unpartition().to_host_vec().sync_on(&stream)?;
    for head in 0..HEADS {
        for pos in 0..CAPACITY {
            for d in 0..D {
                let expected = if (START..START + ROWS).contains(&pos) {
                    values[((pos - START) * HEADS + head) * D + d].to_f32()
                } else {
                    0.
                };
                assert_eq!(k[(head * CAPACITY + pos) * D + d].to_f32(), expected);
                assert_eq!(v[(head * CAPACITY + pos) * D + d].to_f32(), expected);
            }
        }
    }
    Ok(())
}
