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
