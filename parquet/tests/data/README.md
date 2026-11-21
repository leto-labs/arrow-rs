# Parquet test data

`bitpacking_levels.parquet` is the 2,589-byte parquet-mr 1.13.1 fixture
attached to [arrow-rs issue #5338](https://github.com/apache/arrow-rs/issues/5338#issuecomment-1913604976).
The [writer source](https://github.com/jhorstmann/parquet-writer-example/tree/9e97ddb8e286ee6440b611b19ad1a7c19942dd3d)
selects the Java BIT_PACKED writer for V1 levels. The issue comment records
the Java readback used as the expected row in the regression tests.
SHA-256: `ea00396f49ca9798ca761f3621caa64858a0db7a8191d06f77e9a7b9a60a5d3d`.
