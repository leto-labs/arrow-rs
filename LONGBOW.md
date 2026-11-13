# Longbow Arrow fork notes

FUG, PIO and NEX are UPSTREAMABLE patch families for
NATIVE-PARQUET-DECODE. They change Parquet's native dictionary and flat nullable
primitive readers; CPU-specific kernels remain behind checked runtime dispatch.

NEX initializes NULL value slots to zero. The previous padding path could leave
stale values in these slots. The validity bitmap and every valid value are
identical. Arrow consumers must continue to use validity when interpreting a
nullable slot; physical bytes in NULL slots are not a logical value.

The shared native layout predicate admits exactly i32, i64, f32 and f64.
Kernel size assertions document its four/eight-byte contract. Unit properties
run scalar, AVX2 and AVX-512 tiers independently on supporting CPUs through a
thread-local test-only override; overrides never bypass feature checks and are
absent from production builds. The AVX2 tier uses portable NULL expansion.
Widths 18–24 include full dictionaries and valid indices through bit 23.

## PEH: Parquet error handling

PEH is an UPSTREAMABLE patch family for PARQUET-INVALID-INPUT. A file with
invalid page headers, level data or value encodings gives a `ParquetError`.
Before, some of these files made the reader panic, and two kinds made it run
for a long time or without end:

- A skipped Thrift collection with a corrupt element count ran once for each
  declared element after the input ended, because a skip at the end of a `Read`
  input was not an error.
- `skip_records` did not end when the repetition levels of a page ended before
  its declared value count.

One location was an unchecked slice: after a PLAIN byte array `skip` past the
page, `ByteViewArrayDecoderPlain::read` called `get_unchecked` with a range
outside the buffer. A build with debug assertions aborts there.

The checks run once for each page, run or batch. No check is added for each
value, except in the PLAIN byte array `skip`, where the existing length read
now also compares the end offset with the buffer length. Results for valid
files do not change. Some invalid files that the reader accepted now fail:

- a dictionary index bit width above 32 in a byte array page (the primitive
  path already rejected it);
- a negative DELTA_BYTE_ARRAY length;
- a negative count in the header of a page that a row selection skips;
- a BYTE_STREAM_SPLIT page with fewer stored values than the page needs (the
  decoder gave values from the wrong bytes);
- in a selected read, repetition levels that give no level before the value
  count of the page is reached (the full read already failed).

`BitReader::get_vlq_int` returns `None` for an encoding longer than ten bytes;
it panicked before. `RleDecoder` keeps the low 32 bits of a run length, as the
release build did before; the checked build no longer overflows.

The regression files are in `parquet/tests/arrow_reader/invalid_files/`, with
their origin in its `README.md`. Unit tests of the family are in modules named
`invalid_input_tests`.

Lane PR2 added one location, which files of other writers reach: pyarrow
writes a dictionary for `FIXED_LEN_BYTE_ARRAY` columns. A dictionary key
outside the dictionary indexed the dictionary page past its end and panicked.
It gives `invalid key=N for dictionary`, as the byte array readers do. A
dictionary-encoded page of such a column without a dictionary page gives
`missing dictionary page for column`.

## PAB: Parquet allocation bounds

PAB is an UPSTREAMABLE patch family for PARQUET-INVALID-INPUT. A count, a
length or a size that the reader takes from a page header or from an encoded
block is compared with what the page can hold before the reader allocates
memory for it. A count that the page cannot hold gives a `ParquetError`.
Before, the reader allocated for the declared count. The Rust runtime ends the
process when an allocation fails, and no caller can report that as an error.

The limits are:

- The length blocks of DELTA_LENGTH_BYTE_ARRAY and DELTA_BYTE_ARRAY (prefix
  and suffix lengths): at most the value count of the page, and at most the
  blocks that the remaining bytes of the page can hold (each block has at
  least a minimum delta and one bit width for each mini block).
- A dictionary page: at most the PLAIN values that its bytes can hold (one bit
  for a BOOLEAN, four bytes for a BYTE_ARRAY length, the type length
  otherwise). A byte array dictionary that holds fewer values than its header
  declares is an error.
- A data page: at most the value count of its column chunk.
- Snappy and LZ4 pages: the uncompressed size is at most 22 times (Snappy) or
  255 times (LZ4) the compressed size, the largest expansion of each format.
  The Snappy stream length must equal the size in the page header; before, a
  shorter stream left zeros at the end of the page without an error.
- GZIP, Brotli and LZ4 frame pages: the decoder stops one byte after the size
  in the page header. The Zstandard decoder writes at most that size.
  `decode_page` reserves no memory; each codec reserves after its check.
- `FIXED_LEN_BYTE_ARRAY` values: the type length is from the file schema and
  can be 2^31 - 1. A capacity hint of that length for each row is dropped when
  it cannot be allocated, and the slots for NULL values are an error then.

The checks run once for each page or dictionary. No check runs for each
value. Results for valid files do not change. `DeltaByteArrayDecoder::skip`
of the generic decoders allocates for the values that are left, not for the
count that the caller names.

The regression files are in `parquet/tests/crafted_files/`, with the scripts
that write them. `parquet/tests/allocation_bounds.rs` reads them with an
allocator that refuses large allocations. Unit tests of the family are in
modules named `allocation_bound_tests`.

A file can declare the same large count in its footer, its page header and
its encoded block. Such a file is consistent, and the reader allocates for
its count: a length block of 2^31 - 1 lengths needs 8 GiB. PAB does not limit
that.

## PCV: page checksum verification

PCV is a KEEP patch family for PARQUET-INVALID-INPUT. The `crc` feature is in
the default features, so the reader verifies the CRC-32 of each page that
carries one, before the page is decompressed or decoded. A mismatch gives
`Page CRC checksum mismatch`, with the page type, the page length and both
checksums. A page without a checksum is read as before. A page that a row
selection skips is not read and not verified.

The feature also enables `crc32fast/std`, which selects the carry-less
multiplication kernel at run time. Without it, a build that does not target
that CPU feature used the table kernel: verification then took 4.8% of the
decode time of TPC-H SF1 `lineitem` with checksums. With it, verification
adds 0.2% to the instructions and no measurable time (Longbow report
`docs/research/parquet-decode-bounds-20261113.md`).

The regression files are in `parquet/tests/crafted_files/checksums/`;
`parquet/tests/arrow_reader/checksum.rs` reads them.

## PCW: page checksum writing

PCW is an UPSTREAMABLE family for PARQUET-RESULT-INTEGRITY (lane PW1).
With the `crc` feature, `WriterPropertiesBuilder::set_write_page_checksum`
enables CRC-32 on every data and dictionary page. It defaults to false so
Longbow generation retains its previous bytes and file sizes. Longbow result
writers enable it through their shared properties builder.

Both the serialized page writer and the buffered Arrow page writer hash the
compressed payload before encryption. They attach the checksum before
serializing the page header. Header sizes and downstream page offsets include
the additional field. The existing `crc32fast/std` feature selects the runtime
hardware kernel. No row value, encoding or compression policy changes.

`file::page_checksum_tests` checks the raw header CRC against each payload,
DataPage V1/V2, dictionary and plain encodings, supported compression samples,
property round trips and the checksum-free default. Longbow adds result and
parallel append corruption tests. PCV verifies these pages on read.
