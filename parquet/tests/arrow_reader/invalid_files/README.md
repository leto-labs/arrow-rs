# Invalid Parquet files

Each file made the Parquet reader panic or loop at Arrow 59.3.2 and at the
Longbow fork commit `bf76fe14`. With the PEH patch family the reads below
return a `ParquetError`. `invalid_files.rs` reads every file.

## Origin

The files are mutations of valid files. The Longbow Z2 review wrote 64 valid
files with nullable, required, list and struct columns in several encodings,
then changed a few bytes of the page data of each file. The footer is valid in
every file. A mutation is identified by the mutation seed, the base file and
the mutation number; `docs/research/parquet-reader-error-handling-20261112.md`
in the Longbow repository records the program and how to make a file again.

Each `NAME.parquet.selection` file holds the row selection of the selected
read: `select` or `skip` for the first run, then the run lengths. The full read
uses a batch size of 1024. The selected read uses a batch size of 97.

## Files

"Location" is the panic location in the fork commit `bf76fe14` with overflow
checks, or the function that did not end.

| File | Bytes | Seed, base file, mutation | Location before PEH | Invalid content |
|---|---:|---|---|---|
| `def_levels_packed_run_past_end.parquet` | 5098 | 4242 f002 218 | `arrow-buffer/src/util/bit_chunk_iterator.rs:224` (offset + len out of bounds) from `PackedDecoder::read` | A bit-packed definition level run declares more groups than the page holds. |
| `def_levels_packed_run_past_end_skip.parquet` | 5098 | 9 f002 285 | `arrow-buffer/src/util/bit_chunk_iterator.rs:57` (range end out of range) from `PackedDecoder::skip` | The same, in the skip of a selected read. |
| `def_levels_packed_run_no_data.parquet` | 5098 | 3 f002 55 | `arrow-buffer/src/util/bit_util.rs:137` (slice must not be empty) from `PackedDecoder::read` | A bit-packed definition level run header is the last byte of the level data. |
| `def_levels_packed_run_short_data.parquet` | 8584 | 1 f000 44 | `arrow-buffer/src/util/bit_util.rs:142` (slice is too small) from `PackedDecoder::read` | A bit-packed definition level run is longer than the data, in a later batch. |
| `v2_levels_longer_than_page.parquet` | 6559 | 1 f008 94 | `bytes/src/bytes.rs:374` (range end out of bounds) from `column/reader.rs:556` | The level lengths in a V2 page header exceed the page data. |
| `plain_byte_array_length_past_end.parquet` | 225354 | 10 f059 134 | `parquet/src/arrow/array_reader/byte_array.rs:384` (subtract with overflow) | A PLAIN byte array length points past the page; `skip` moved the offset past the buffer. |
| `plain_byte_array_no_values_declared.parquet` | 7546 | 2 f054 206 | `parquet/src/arrow/array_reader/byte_array.rs:391` (divide by zero) | The dictionary page of a string column declares zero values and holds value bytes. |
| `delta_byte_array_negative_suffix_length.parquet` | 60868 | 4242 f034 55 | `parquet/src/arrow/decoder/delta_byte_array.rs:95` (read) and `:126` (skip): add with overflow | A DELTA_BYTE_ARRAY suffix length is negative. |
| `dictionary_indices_empty.parquet` | 66719 | 2 f021 242 | `parquet/src/arrow/decoder/dictionary_index.rs:46` (index out of bounds) | A dictionary-encoded data page has no value bytes, so no bit width byte. |
| `v1_page_type_without_header.parquet` | 5098 | 1 f002 25 | `parquet/src/column/page.rs:365` (unwrap on None) | The page type is DATA_PAGE and the V1 header is absent. |
| `v2_page_type_without_header.parquet` | 6559 | 2 f008 64 | `parquet/src/column/page.rs:378` (unwrap on None) | The page type is DATA_PAGE_V2 and the V2 header is absent. |
| `dictionary_data_page_without_dictionary.parquet` | 6715 | 3 f049 154 | `parquet/src/column/reader/decoder.rs:236` (expect) | A dictionary-encoded data page has no dictionary page before it. |
| `delta_binary_packed_zero_values.parquet` | 60868 | 4242 f034 169 | `parquet/src/encodings/decoding.rs:840` (subtract with overflow) | A DELTA_BINARY_PACKED header declares zero values in a page that has values. |
| `byte_stream_split_short_data.parquet` | 7546 | 8 f054 170 | `parquet/src/encodings/decoding/byte_stream_split_decoder.rs:61` (read) and `:58` (after a skip): index out of bounds | A BYTE_STREAM_SPLIT page holds fewer values than its header declares. |
| `rle_levels_run_length_overflow.parquet` | 6559 | 10 f008 70 | `parquet/src/encodings/rle.rs:742` (multiply with overflow) | A bit-packed run header in RLE definition levels declares 2^60 groups or more. |
| `dictionary_index_width_148.parquet` | 5098 | 7 f002 148 | `parquet/src/util/bit_util.rs:256` (range end out of range) from `RleDecoder::reload` | The dictionary index bit width byte is 148. |
| `dictionary_index_width_142.parquet` | 5098 | 6 f002 270 | `parquet/src/util/bit_util.rs:705` (num_bits assertion) from `RleDecoder::get_batch` | The dictionary index bit width byte is 142. |
| `dictionary_index_width_90.parquet` | 6715 | 6 f049 255 | `parquet/src/util/bit_util.rs:705` (read) and `:789` (skip): num_bits assertions | The dictionary index bit width byte is 90. |
| `rle_run_header_too_long.parquet` | 6715 | 3 f049 201 | `parquet/src/util/bit_util.rs:876` (shift left with overflow) from `RleDecoder::reload` | An RLE run header is a varint of more than ten bytes. |
| `page_header_map_past_end.parquet` | 6559 | 4242 f008 25 | loop: `ThriftCompactInputProtocol::skip_till_depth` (map branch) | A page header has an unknown map field whose element count is larger than the file. |
| `repetition_levels_end_early.parquet` | 6559 | 4242 f008 216 | loop: `GenericColumnReader::skip_records`, selected read only | The repetition levels of a page end before its declared value count. |

The selected read of `def_levels_packed_run_past_end.parquet` skips the
invalid page and succeeds. Every other read of every file fails.
