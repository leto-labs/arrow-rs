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
| `dictionary_strings_count_i32_max` | 248 | strings, dictionary | The dictionary page header declares 2^31 - 1 values; the page holds 2. | Reserves an offset (8 GiB or 16 GiB) or a view (32 GiB) for each declared value. As offsets the read then returned rows. |
| `dictionary_int96_count_i32_max` | 259 | `INT96`, dictionary | The same, for a type that the generic dictionary decoder reads. | Allocates and fills 12 bytes for each declared value: 24 GiB. |
| `data_page_count_above_chunk` | 247 | `INT64`, PLAIN | The data page header declares 2^31 - 1 values; the column chunk has 4. | No allocation. The page count is the limit for the checks above, so it has its own limit. |
| `snappy_size_i32_max` | 260 | `INT64`, Snappy | The page header and the Snappy stream declare 2^31 - 1 uncompressed bytes; the stream holds 32. | Fills 2 GiB with zeros before it decodes. |
| `lz4_raw_size_i32_max` | 256 | `INT64`, LZ4_RAW | The page header declares 2^31 - 1 uncompressed bytes. | The same. |
| `gzip_longer_than_declared` | 32850 | `INT64`, GZIP | The page header declares 32 uncompressed bytes; the stream holds 32 MiB. | Reads the stream to its end, then compares the lengths. |

Valid files: `valid_delta_length`, `valid_delta_byte_array`,
`valid_dictionary_strings`, `valid_dictionary_int96`, `valid_snappy`,
`valid_lz4_raw`, `valid_gzip`.

## `type_length/`: a valid file that Arrow cannot hold

`fixed_len_type_length_i32_max` is valid: its one column `v` is an optional
`FIXED_LEN_BYTE_ARRAY` of 2^31 - 1 bytes and its 100,000 values are NULL. Arrow
needs a slot of the type length for each NULL value. Before PAB the reader
reserved the type length for each row of a batch before it read a page (2 TiB
for a batch of 1024 rows), and the failed allocation ended the process. With
PAB the capacity hint is dropped when it cannot be allocated, and the slots of
the NULL values are an error when the host cannot provide them.
