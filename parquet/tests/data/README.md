# Legacy V1 BIT_PACKED test data

`legacy_bit_packed.parquet` is generated from literal sample values by
`make_legacy_bit_packed.rs` using Parquet's low-level serialized page writer.
Both the generator and its output are licensed under Apache-2.0.
The fixture's first row has the scalar values reported by the parquet-mr
readback. [arrow-rs issue #5338](https://github.com/apache/arrow-rs/issues/5338#issuecomment-1913604976)

The `optional_double` chunk starts with an MSB-first BIT_PACKED definition
page and continues with an RLE definition page. The `outer_items.inner_items`
chunk uses two-bit repetition and definition levels. Its first record spans
both legacy pages.

From the Longbow checkout, regenerate the file with:

```sh
scripts/vendored-tests.py parquet --test legacy_bit_packed --run-ignored only -E 'test(regenerate_fixture)'
```

The regular fixture test compares the checked-in bytes with the generator.
