# Crafted Parquet files

Small files that a script writes byte by byte. A Parquet writer does not
produce them: each invalid file has one page header field or one encoded count
that its page contradicts. The Longbow patch family PAB (allocation bounds)
uses them as regression files.

## Scripts

The scripts use only the Python standard library (Python 3.9 or later) and
write the same bytes each time:

```sh
cd parquet/tests/crafted_files
python3 make_allocation_bounds_files.py   # allocation_bounds/, type_length/
```

`parquet_file.py` encodes the Thrift compact protocol, the page headers, the
value encodings and the footer. Every file has one row group. The files of
`allocation_bounds/` have four rows and a valid first column `e` (`INT64`,
required, PLAIN, the values 0 to 3).

## `allocation_bounds/`: a count that the page cannot hold

Read by `tests/allocation_bounds.rs`. The test program replaces the allocator
with one that refuses an allocation above 64 MiB during a read, as a host
without that memory does. The Rust runtime ends the process when an allocation
fails, so before PAB the reads of these files end the test process with
`SIGABRT` (`memory allocation of 4398046511104 bytes failed`). With PAB each
read returns a `ParquetError` and no allocation of the read is above 1 MiB.

Column `v` holds the invalid page. Each invalid file has a valid file that
differs only in the declared count.

| File | Bytes | Column `v` | Declared | Without PAB |
|---|---:|---|---|---|
| `delta_length_count_2e40` | 238 | strings, DELTA_LENGTH_BYTE_ARRAY | The length block declares 2^40 lengths; the page has 4 values. | Allocates 4 bytes for each declared length: 4 TiB. |
| `delta_byte_array_prefix_count_2e40` | 242 | strings, DELTA_BYTE_ARRAY | The prefix length block declares 2^40 lengths. | The same. |
| `delta_byte_array_suffix_count_2e40` | 242 | strings, DELTA_BYTE_ARRAY | The suffix length block declares 2^40 lengths. | The same. |
| `delta_length_mini_blocks_2e56` | 234 | strings, DELTA_LENGTH_BYTE_ARRAY | A length block of one value under a header of block size 2^61 and 2^56 mini blocks of 32 values. A block of one value stores no block, so no check compares the header with the page bytes. | Reserved one byte for each declared mini block width while it looked for the end of the block: 64 PiB. |
| `delta_byte_array_mini_blocks_2e56` | 254 | strings, DELTA_BYTE_ARRAY | The same header on the prefix and the suffix lengths. | The same. |
| `dictionary_strings_count_i32_max` | 248 | strings, dictionary | The dictionary page header declares 2^31 - 1 values; the page holds 2. | Reserves an offset (8 GiB or 16 GiB) or a view (32 GiB) for each declared value. As offsets the read then returned rows. |
| `dictionary_int96_count_i32_max` | 259 | `INT96`, dictionary | The same, for a type that the generic dictionary decoder reads. | Allocates and fills 12 bytes for each declared value: 24 GiB. |
| `snappy_size_i32_max` | 260 | `INT64`, Snappy | The page header and the Snappy stream declare 2^31 - 1 uncompressed bytes; the stream holds 32. | Fills 2 GiB with zeros before it decodes. |
| `lz4_raw_size_i32_max` | 256 | `INT64`, LZ4_RAW | The page header declares 2^31 - 1 uncompressed bytes. | The same. |
| `gzip_longer_than_declared` | 32850 | `INT64`, GZIP | The page header declares 32 uncompressed bytes; the stream holds 32 MiB. | Reads the stream to its end, then compares the lengths. |

Files whose page declares 2^31 - 1 values and holds a few. No allocation
follows the declared count: the full read fails when the values need more
bytes than the page holds, and a selected read returns its rows.

| File | Bytes | Column `v` | Declared | Without the fix |
|---|---:|---|---|---|
| `data_page_count_above_chunk` | 247 | `INT64`, PLAIN | The data page header declares 2^31 - 1 values; the column chunk has 4. | No allocation: the PLAIN decoder reads into batch buffers. A page may declare more values than its chunk, because some writers under-report the chunk count (`valid_chunk_count_below_pages`). |
| `delta_length_zero_width_count_i32_max` | 243 | strings, DELTA_LENGTH_BYTE_ARRAY | The page and its length block declare 2^31 - 1 lengths of 1; the column chunk has 4. The block size is 2^30 and every bit width is 0, so ten bytes hold the lengths. | Decodes every length when the page is set: 8 GiB. |
| `delta_length_zero_width_chunk_count_i32_max` | 255 | strings, DELTA_LENGTH_BYTE_ARRAY | The same page in a column chunk and row group that also declare 2^31 - 1 values: the footer agrees with the page, and a page of 2^31 - 1 empty strings would be valid. | The same: 8 GiB. |
| `delta_byte_array_zero_width_chunk_count_i32_max` | 279 | strings, DELTA_BYTE_ARRAY | Prefix and suffix length blocks of 2^31 - 1 values in such a chunk. | Decodes every prefix and suffix length: 16 GiB. |

The DELTA_LENGTH_BYTE_ARRAY and DELTA_BYTE_ARRAY decoders now decode the
lengths 1024 at a time as the values are read. The bytes of a valid page of
empty strings cannot bound its count, so no count check can reject these
files without rejecting valid ones.

Valid files: `valid_delta_length`, `valid_delta_byte_array`,
`valid_dictionary_strings`, `valid_dictionary_int96`, `valid_snappy`,
`valid_lz4_raw`, `valid_gzip`, and `valid_chunk_count_below_pages`: a lax
writer's chunk that declares two values for its four, which reads its values.

## `type_length/`: a valid file that Arrow cannot hold

`fixed_len_type_length_i32_max` is valid: its one column `v` is an optional
`FIXED_LEN_BYTE_ARRAY` of 2^31 - 1 bytes and its 100,000 values are NULL. Arrow
needs a slot of the type length for each NULL value. Before PAB the reader
reserved the type length for each row of a batch before it read a page (2 TiB
for a batch of 1024 rows), and the failed allocation ended the process. With
PAB the capacity hint is dropped when it cannot be allocated, and the slots of
the NULL values are an error when the host cannot provide them.
