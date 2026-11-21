# Longbow 60.1 patch branch

Base: Apache Arrow tag `60.1.0`, commit
`b391e5af7dff9f6909af45c01f7dddb68ba08c0f`.

Published sources are the Apache `60.1.0` Git tag and the 60.1.0 crate archives
linked below. [LICENSE.txt](LICENSE.txt) and
[NOTICE.txt](NOTICE.txt) remain unchanged. Only `arrow-arith`,
`arrow-select` and `parquet` carry patches. All other resolved Arrow packages
still come from this same gitlink. The consuming Longbow repository records the complete cross-engine inventory
in `third_party/README.md` and `docs/UPSTREAM_MATRIX.toml`.

### arrow-arith 60.1.0

- Published source: [arrow-arith 60.1.0](https://static.crates.io/crates/arrow-arith/arrow-arith-60.1.0.crate); [Apache tag source](https://github.com/apache/arrow-rs/tree/60.1.0/arrow-arith).
- Local source: `arrow-rs/arrow-arith`.

| Family | Purpose in this stack | Dependent feature | Classification |
|---|---|---|---|
| DK | Optimistic checked decimal add, subtract and multiply with the checked kernel as fallback | NARROW-DECIMALS | UPSTREAMABLE |

### arrow-select 60.1.0

- Published source: [arrow-select 60.1.0](https://static.crates.io/crates/arrow-select/arrow-select-60.1.0.crate); [Apache tag source](https://github.com/apache/arrow-rs/tree/60.1.0/arrow-select).
- Local source: `arrow-rs/arrow-select`.

| Family | Purpose in this stack | Dependent feature | Classification |
|---|---|---|---|
| R | Push borrowed record-batch ranges beside upstream interleaving with equivalent bypass and view-GC policy | RX-REPARTITION | UPSTREAMABLE |

### parquet 60.1.0

- Published source: [parquet 60.1.0](https://static.crates.io/crates/parquet/parquet-60.1.0.crate); [Apache tag source](https://github.com/apache/arrow-rs/tree/60.1.0/parquet).
- Local source: `arrow-rs/parquet`.

| Family | Purpose in this stack | Dependent feature | Classification |
|---|---|---|---|
| DCX | Read flat primitive chunk dictionaries, borrowed hybrid runs and selected nullable keys | DICTIONARY-CODE-EXECUTION | UPSTREAMABLE |
| DV | Apply static Arrow predicate verdicts to selected Parquet rows using DCX index runs | DICTIONARY-CODE-EXECUTION | UPSTREAMABLE |
| LMS | Materialize selected flat dictionary byte-array views with DCX ranks and reuse selected fixed-width reads | LATE-MATERIALIZED-SCAN | UPSTREAMABLE |
| DPC | Hook decompressed page caching and atomic page leases into PageIndexProvider and shared fetch planning | SCAN-PAGE-CACHE | UPSTREAMABLE |
| DRL | Decode dictionary runs and definition levels with register cursors, SIMD lookup and in-place NULL expansion | NATIVE-PARQUET-DECODE | UPSTREAMABLE |
| NPI | Classify repeated-leaf null pages by value count to preserve valid column indexes | DATA-SKIPPING | UPSTREAMABLE |

NPI covers two cases in one `List<Int64>` row group. A page containing
`[NULL, NULL]` retains its column index and records two null entries. A page
containing `[NULL, 7]` records its minimum and maximum as 7. The fix uses the
same level-entry denominator as `null_count`. It does not change the offset
index's row accounting and has no Longbow-specific dependencies.
| FUG | Fuse primitive dictionary unpack and gather with runtime AVX2 and AVX-512 dispatch | NATIVE-PARQUET-DECODE | UPSTREAMABLE |
| NEX | Decode flat nullable primitives into logical output slots with zero-filled NULL slots | NATIVE-PARQUET-DECODE | UPSTREAMABLE |
| PAB | Bound page and encoded-block allocations by what their input can hold, including decompression | PARQUET-INVALID-INPUT | UPSTREAMABLE |
| PCV | Verify page CRC-32 checksums by default with runtime checksum dispatch | PARQUET-INVALID-INPUT | KEEP |
| PCW | Write opt-in page CRC-32 checksums over compressed data and dictionary pages | PARQUET-RESULT-INTEGRITY | UPSTREAMABLE |
| PEH | Return errors for invalid page data instead of panics or loops; upstream owns two former checks | PARQUET-INVALID-INPUT | UPSTREAMABLE |
| Q | Stream the runs of a mask-backed selection in the selector cursor and remember a partial run's end | NATIVE-PARQUET-SCAN | UPSTREAMABLE |
| PIO | Append primitive dictionary values through spare capacity while preserving initialized-prefix and failed-state rules | NATIVE-PARQUET-DECODE | UPSTREAMABLE |
| U | Enable runtime SIMD UTF-8 dispatch in portable builds with scalar-equivalence properties | STRING-UTF8-VALIDATION | UPSTREAMABLE |
| Z1-A | Borrow aligned immutable primitive dictionary page bytes after fixed-length normalization | NATIVE-PARQUET-DECODE | UPSTREAMABLE |
| Z1-B | Validate complete primitive ID batches and dispatch dictionary gathers through AVX2 | NATIVE-PARQUET-DECODE | UPSTREAMABLE |

## Decode and page-cache notes

LMS extends the opt-in `ArrayReader::read_selected` path to flat byte-array
columns read as Utf8View or BinaryView. It covers RLE_DICTIONARY and
PLAIN_DICTIONARY data pages with required or hybrid-coded flat nullable
levels. It uses the DCX run cursor and the existing validity bitmap from
definition levels, then gathers the selected dictionary views by their
physical ranks. The returned views share the raw dictionary buffers.
UTF-8 validation checks the selected entries. Byte-array pages with other
encodings or legacy BIT_PACKED levels use full decode and compaction.
The existing DCX fixed-width path covers INT32, INT64, FLOAT and DOUBLE
dictionary and PLAIN pages, including date and Decimal64 conversions.

On the LMS byte-view path, an unselected data page with a known row span
can be skipped across vector boundaries without loading its body. The
serialized page reader fills a skipped compressed page when the file cache
admits it. Ordinary byte-view reads keep their existing decoder, including
when selected mode hands the active page back to it. The DV row-filter plan
continues to use its own dispatch.

DV evaluates a flat column's dictionary once for each participating predicate and
chunk selection. It uses the normal Arrow column reader to interpret PLAIN entries
with the column's logical type and uses the DCX index walker for data pages. The
NULL verdict is applied through the page's definition levels. Before any verdict
work, DV checks the page counts of every chunk up to the end of the selected
span: each unindexed page header, and each indexed page that holds selected
rows. A DataPage v2 row count that differs from its value count, an index span
that differs from a decoded page, or spans that do not add up to the row group
send the whole read unit to the value reader, which defines how such a file
reads. Mixed pages and entry or cursor errors use the existing value reader.
Verdict bitmaps cover only the span from the first to the last selected row. A
selection covering less than one quarter of that span bypasses them before
anything is allocated. Ordinary row filters
use the value plan directly. In verify mode, the value predicate must select
exactly the same rows wherever a verdict was used.

DCX keeps code readers opt-in. The flat primitive reader owns the dictionary
and checks the page encoding in its own path. On worker-01 the three hot PLAIN
UTF-8 functions retained the same instructions and call targets after
relocation normalization. A same-source linker-layout control moved the
synthetic case by 4.32%, and the isolated DCX comparison moved it by 1.26%.

The portable `vertical_ceiling --groups codes` ABBA on the three
`store_sales.1` columns in task 5.11a measured 30.32 user cycles per row
against the 9-cycle ceiling. In diagnostic cycle samples, the shared cursor
accounted for 55%, packed-specific work for 18%, and repeated-specific work
for 7%. The remaining samples covered shared definition-level handling and
other scan work. See the lane's benchmark evidence in the consuming
repository's benchmark data.

The walker calls each sink directly. Its dense verdict sink checks packed
codes in the lookup table through width 20 and tracks the maximum code at
wider widths. Packed validity is built in row order with one padded
little-endian load per 64 levels. On the AVX-512 tier with BMI2 and POPCNT,
row expansion uses hardware bit deposit. Errors invalidate the caller's
scratch output.

Selected reads of flat INT32, INT64, FLOAT and DOUBLE chunks use the same
run walker. The reader keeps its current run boundary and reuses rank,
validity and code buffers between vectors. It validates selected packed codes
and skips unselected runs by their headers. The dictionary retains its PLAIN
page bytes and can gather unaligned little-endian values directly. Fixed-width
PLAIN data pages use the same gather. Legacy levels and other value encodings
use the existing decoders. Unselected data pages are stepped over separately,
with compressed bodies decoded only when the page cache offers a fill ticket.

Before growing a bitmap, the reader checks that run lengths and payloads
cover the declared count. A final packed group may pad up to seven slots.
The reader also accepts trailing zero-byte padding emitted by fastparquet.
V2 flat pages require equal row and value counts. The ordinary reader and
both DCX paths share a most-significant-bit-first decoder for legacy V1
BIT_PACKED level streams. DCX supports only flat columns, so only definition
levels use that decoder on DCX paths.
[Parquet encodings](https://parquet.apache.org/docs/file-format/data-pages/encodings/) ·
[Parquet Thrift headers](https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift)

FUG requires Parquet MSRV 1.89 for stable AVX-512 intrinsics; other packages keep
their upstream minimums. SIMD paths select instructions at run time and have
scalar differential coverage. Z1-A borrows only aligned immutable dictionary
bytes after fixed-length normalization. PIO keeps initialized-prefix and
failed-state rules; NEX writes zero into NULL slots without changing validity or
valid values. DRL replaces the earlier dense NEX window with run cursors,
word-wise definition levels and in-place NULL expansion. See the historical
decode-floor report (in the consuming Longbow repository) and
dictionary report (in the consuming Longbow repository) for the original
screens; they are not measurements of the new stack.

PEH returns errors for invalid encodings, PAB bounds allocations and decompression
by available input, PCV verifies supplied checksums by default, and PCW writes
checksums only when requested. Valid-file results remain unchanged. Retained
regression fixtures are under
[invalid_files](parquet/tests/arrow_reader/invalid_files/README.md) and
[crafted_files](parquet/tests/crafted_files/README.md).
Input-size allocation bounds do not impose per-scan work limits: extremely large
zero-width runs can still take substantial decode work.

DPC is a hook for an engine-owned decompressed page cache; Parquet owns neither
the cache budget nor eviction. The hook is attached to metadata for a file
generation and reworked onto Arrow 60's `PageIndexProvider` and shared fetch
planning. Complete column residency can omit compressed, unencrypted fetches;
partial, uncompressed and encrypted cases keep the cold path. Atomic page leases
keep planned hits resident through decode and reclamation, with memory ownership
in Longbow. Failed hit checks fall back to cold reads. Cached pages have already
passed header and checksum checks; cache identity, integrity and lease lifetime
are the caller's obligations. Without a cache handle the normal reader path is
used. See ADR 0053 (in the consuming Longbow repository).

## Families absent from this stack

- L: upstream removed the `parquet-variant-json` dependency, so the manifest
  exemption is unnecessary.
- V: upstream `ParquetPushDecoder::scan_plan()` replaces the advisory plan;
  the consuming engine owns read-ahead policy and reader advice.

PEH uses upstream overflow and dictionary-width checks (#11517, #10725)
instead of the two old hunks. The other invalid-input error paths remain.

## Family notes

Carried from `longbow-59.3.2` and checked against this stack; the families
keep their purpose on 60.1.

FUG, PIO, NEX and DRL are UPSTREAMABLE patch families for
NATIVE-PARQUET-DECODE. They change Parquet's native dictionary and flat nullable
primitive readers; CPU-specific kernels remain behind checked runtime dispatch.
DRL (below) replaced NEX's dense decode window and forward expansion with one
dense decode and an in-place backward expansion; NEX's zero NULL slots remain.

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

### PEH: Parquet error handling

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

### PAB: Parquet allocation bounds

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

### PCV: page checksum verification

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

### PCW: page checksum writing

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

### DPC: decompressed page cache hook

DPC is an UPSTREAMABLE patch family for SCAN-PAGE-CACHE (Longbow ADR 0053).
The crate holds no pages. It defines `parquet::file::page_cache::PageCache`
and uses it in two places:

- `SerializedPageReader` looks a page up by the file offset of its header
  before it reads, checks and decompresses it, and offers the decoded `Page`
  to the cache after a miss. `peek_next_page` and `skip_next_page` step over a
  resident page by its stored length, without its file bytes.
- The push decoder's fetch planning (`InMemoryRowGroup::fetch_ranges`) leases
  the pages that a row group read needs with one `PageCache::lease` call for
  each decoding stage, or takes them from the caller's lease
  (`ParquetPushDecoderBuilder::with_page_lease`) with no cache call. A column
  chunk whose needed pages are all leased is not fetched
  (`ColumnChunkData::Cached`). The reader holds the lease, so an eviction
  between planning and decoding cannot remove a page whose bytes were not
  fetched.

The engine attaches one handle to the `ParquetMetaData` of a file generation
(`ParquetMetaData::with_page_cache`), and owns the memory, the eviction and
the identity of the file. The handle takes no part in metadata equality. A
copy of the metadata, also through `ParquetMetaDataBuilder` or a later page
index load, keeps it.

Limits that the tests state:

- Only compressed, unencrypted column chunks use the cache. The buffer of an
  uncompressed page is a slice of the fetched chunk, so caching it would keep
  the chunk alive and save no decompression. A reader that decodes page
  statistics (`ReaderProperties::read_page_stats`) does not use it either.
- A cached page passed the reader's header parse and CRC when it was filled.
  A hit repeats neither, but it repeats the placement checks (codec of the
  column chunk, fit in the chunk, offset-index length and page kind); see the
  module documentation of `parquet/src/file/page_cache.rs`. On 60.1 the
  family plans from `PageIndexProvider` and the push decoder's `scan_plan`.
- A lease set's pages cover its ranges without a gap, each followed to the
  next by its stored length; with an offset index each range is one page
  whose stored length is its location's. Without an offset index, a column
  chunk is leased only when the resident pages cover it without a gap. After a selective read that stepped over
  pages, such a chunk is fetched again and its resident pages are still hits.
- The synchronous `SerializedFileReader` row group reader does not pass the
  handle. Longbow's scans read through the push decoder.

With no handle attached nothing changes. Results never change.

### DRL: dictionary runs and definition levels

DRL is an UPSTREAMABLE patch family for NATIVE-PARQUET-DECODE (Longbow scale-up
task 5.10, lane V4). It changes how the native primitive reader decodes
dictionary indices and the definition levels of flat nullable columns. Results
do not change; NULL slots stay zero, as NEX defined them.

- **Run loop** (`encodings/rle/runs.rs`). The dictionary decoder of `i32`,
  `i64`, `f32` and `f64` reads run headers, repeated values and packed runs
  from one borrowed slice with the cursor in a register, instead of one
  `BitReader` call for each header and value. A repeated run of at most 16
  values is one block of vector stores when the output has room; the slots
  after the run are overwritten by the next values or lie outside the
  initialized prefix. A packed run goes to the fused kernels. A cursor inside
  a packed group (after a read that ended mid-run) decodes only to the end of
  the group, so the rest of the run stays on the fused path.
- **Every bit width is fused.** The AVX-512 VBMI kernel unpacks widths 25 to 32
  into 64-bit lanes; widths 1 to 24 as before. It also takes a last block of
  eight values, so a run of 504 values no longer leaves eight for the scalar
  path. On AVX-512 hosts AVX2 stays for widths 1 to 16; AVX2-only hosts use
  it for widths 1 to 24; the scalar path covers CPUs without either.
- **Small dictionaries are looked up by permutation**, not gathered: up to 128
  four-byte or 64 eight-byte entries, in eight 64-byte tables, two
  `vpermi2d`/`vpermi2q` per pair and a blend per index bit. IDs are checked
  against the dictionary length before any lookup, so padding entries are
  never selected.
- **Definition levels** (`arrow/record_reader/definition_levels.rs`). The
  bit-width-1 level decoder writes the validity bitmap through one register
  word and stores only whole aligned words, and counts the set bits as it
  writes them (the extra `count_set_bits` pass is now a debug assertion). Runs
  with one-byte headers are decoded from a local cursor.
- **NULL expansion in place** (`column/reader/spaced.rs`). A nullable batch is
  decoded as one dense run of physical values into the output, then expanded
  to logical slots from the last block to the first, with AVX-512 expand-loads
  of 16 or 8 lanes. A dense value never lies after its slot, so the backward
  order overwrites nothing that is still to be moved. This replaces NEX's
  1,024-value dense window and its forward expansion; the window buffer is
  gone.

The `ss_customer_sk` shape (TPC-DS: repeated runs of about 12 IDs, a NULL about
every 22 rows) was dominated by the cost of a run, not by the gather.

The unit properties of the family run every dispatch tier (scalar, AVX2,
AVX-512): `encodings::rle::runs` (headers, values, block stores),
`encodings::rle::fused` (exact invalid IDs for widths 1 to 32, permutation
table-pair boundaries for both value sizes), `column::reader::spaced`
(in-place expansion against a reference, count mismatches) and the
definition-level decoder tests. The page-level characterization
`test_fused_dictionary_widths_pages_and_nulls` adds two run-heavy patterns
(repeated IDs in runs of 1 to 40, with and without NULL runs) to its widths
1 to 32, V1/V2 pages and batch sizes.

DPC follow-up (rle.34): an optional, nonwaiting atomic lease over page
offsets. A supporting engine keeps the entries visible and charged until the
last lease drops, including during pool reclamation or a clear. Leasing
records no access. Longbow uses this to retain pages from prefetch admission
through queued units and decode without replacing a file's metadata or
rebuilding its Arrow schema.

DPC follow-up (rle.64, Longbow ADR 0063 decision 8): `PageCache::lease`
replaces `pin` and `resident_offsets`. One call leases the pages of several
sets of byte ranges (one set for each column chunk) and returns them in a
`PageLease`: the pages themselves, sorted by offset, with a read mark on
each. A set with a page missing leases nothing. A page reader marks a page
when it reads it and the page passed its checks (`get_next_page`,
`take_deferred_dictionary`, including a dictionary that a skip stepped over
without an offset index; not a peek, a skip or a rejected page), one mark
where a `get` counted one read before, and dropping the
last reference hands the marks to the engine's `LeaseRelease`, which unpins
the pages and records the reads under one more acquisition. A leased page
that is not read records no read, also when a decoder is dropped after
planning (review of rle.64, finding 1). The trait's default lease holds the
pages without pins: it walks each set with `probe`, takes each page of a set
that tiles with `get`, which records its read at once (best effort), and
leases a set only when every such `get` hits (`PageLease::is_charged` is
false). `PageCache::holds_leases` (default false)
lets a prefetcher skip such a cache without counting reads. An engine that admits a row group before decoding it
passes its lease to the decoder, and a cached row group takes the cache's
lock twice instead of once for each page and column. `ParquetPushDecoder::
clear_all_ranges` drops the decoder's reference to the caller's lease.
Planning no longer looks a page up in a partially resident column chunk, so
such a chunk no longer counts hits at planning for the pages before its
first missing page and one miss for that page; its reader still counts one
lookup for each page that it reads.

### Q: mask-backed selector runs

Q is an UPSTREAMABLE family for NATIVE-PARQUET-SCAN, kept by task 3.6 as a
resource bound (lane leto-zca.22), not as a measured speed win. With the
Selectors strategy, stock 60.1 expands a mask-backed `RowSelection` into a
`VecDeque<RowSelector>` up front: 1,048,576 selector slots (16 MiB) for a
1,048,576-row alternating mask, growing with the run count. Q streams runs from
the bitmap with upstream's `MaskRunIter` and remembers the end of a run that a
partial batch returned: 56 bytes and no queue, and a long run is discovered
once across 128 partial batches. Both pass the same 2,646-case
selector/queue/mask differential. Default SQL scans reach this cursor only when
the Selectors strategy is chosen (`force_filter_selections` or a selector
policy); no general speed claim is made.
