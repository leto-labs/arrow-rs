# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Writes the regression files of `allocation_bounds/` and `type_length/`.

Each invalid file is a few hundred bytes (one is 32 KB) and declares a count or
a size that its page cannot hold. Each has a valid file next to it that differs
only in that count, so a failure of the invalid file is a failure of the
count and not of the layout. Run this script in its directory to write the
files again; the output is deterministic.
"""

import gzip
import pathlib
import struct

from parquet_file import (
    BYTE_ARRAY, DATA_PAGE, DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY, DICTIONARY_PAGE,
    FIXED_LEN_BYTE_ARRAY, GZIP, INT64, INT96, LZ4_RAW, OPTIONAL, PLAIN, RLE_DICTIONARY, SNAPPY,
    UTF8, Column, delta_binary_packed, int64_column, lz4_literal, parquet_file,
    plain_byte_arrays, snappy_literal, uvarint)

ROWS = 4
I32_MAX = (1 << 31) - 1
# 2^40 values of four bytes are 4 TiB: the allocation fails on any host, and a
# failed allocation ends the process.
HUGE = 1 << 40

# The INT64 values 0 to 3, as PLAIN data: 32 bytes.
INT64_VALUES = b"".join(struct.pack("<q", value) for value in range(ROWS))
# RLE_DICTIONARY indices 0, 1, 0, 1: bit width 1, one bit-packed group.
INDICES = bytes([1, 0x03, 0b1010])


def strings(encoding, data, page_count=ROWS, chunk_count=ROWS):
    column = Column("v", BYTE_ARRAY, converted_type=UTF8, num_values=chunk_count)
    return column.page(DATA_PAGE, data, num_values=page_count, encoding=encoding)


def delta_length(declared_count=None):
    lengths = delta_binary_packed([1, 2, 3, 4], declared_count)
    return strings(DELTA_LENGTH_BYTE_ARRAY, lengths + b"abbcccdddd")


def zero_width_lengths(count, first):
    # Length blocks of 2^30 values in four mini blocks: block size 2^30,
    # value count `count`, the first length `first`, then each block is a
    # minimum delta of 0 and four bit widths of 0. Five bytes hold 2^30
    # lengths equal to `first`.
    blocks = -(-(count - 1) // (1 << 30))
    lengths = uvarint(1 << 30) + uvarint(4) + uvarint(count) + uvarint(2 * first)
    return lengths + bytes([0, 0, 0, 0, 0]) * blocks


def delta_length_zero_width(count, chunk_count=ROWS):
    # The page and its length block declare `count` values; the column chunk
    # declares `chunk_count`. The lengths are 1 and the page holds 4 bytes.
    return strings(DELTA_LENGTH_BYTE_ARRAY, zero_width_lengths(count, 1) + b"abcd",
                   page_count=count, chunk_count=chunk_count)


def delta_byte_array_zero_width(count, chunk_count=ROWS):
    # Prefix lengths 0 and suffix lengths 1, `count` of each; the page holds
    # 4 suffix bytes.
    data = zero_width_lengths(count, 0) + zero_width_lengths(count, 1) + b"abcd"
    return strings(DELTA_BYTE_ARRAY, data, page_count=count, chunk_count=chunk_count)


def huge_header_lengths(first):
    # One length, `first`, under a header of block size 2^61 and 2^56 mini
    # blocks of 32 values. A block of one value has no block, so nothing
    # checks the block layout against the page bytes.
    return uvarint(1 << 61) + uvarint(1 << 56) + uvarint(1) + uvarint(2 * first)


def delta_length_huge_header():
    return strings(DELTA_LENGTH_BYTE_ARRAY, huge_header_lengths(1) + b"a")


def delta_byte_array_huge_header():
    return strings(DELTA_BYTE_ARRAY, huge_header_lengths(0) + huge_header_lengths(1) + b"a")


def delta_byte_array(prefix_count=None, suffix_count=None):
    # "a", "ab", "abc", "abcd"
    prefixes = delta_binary_packed([0, 1, 2, 3], prefix_count)
    suffixes = delta_binary_packed([1, 1, 1, 1], suffix_count)
    return strings(DELTA_BYTE_ARRAY, prefixes + suffixes + b"abcd")


def dictionary(physical_type, values, declared_count=None, **column_arguments):
    column = Column("v", physical_type, num_values=ROWS, **column_arguments)
    column.page(DICTIONARY_PAGE, values, encoding=PLAIN,
                num_values=2 if declared_count is None else declared_count)
    return column.page(DATA_PAGE, INDICES, num_values=ROWS, encoding=RLE_DICTIONARY)


def dictionary_strings(declared_count=None):
    return dictionary(BYTE_ARRAY, plain_byte_arrays([b"x", b"yy"]), declared_count,
                      converted_type=UTF8)


def dictionary_int96(declared_count=None):
    # Nanoseconds of the day, then the Julian day: 1970-01-01 00:00:00 and 00:00:01.
    values = struct.pack("<QI", 0, 2440588) + struct.pack("<QI", 1_000_000_000, 2440588)
    return dictionary(INT96, values, declared_count)


def compressed(codec, data, uncompressed_size=None):
    column = Column("v", INT64, codec=codec, num_values=ROWS)
    return column.page(DATA_PAGE, data, num_values=ROWS, encoding=PLAIN,
                       uncompressed_size=len(INT64_VALUES) if uncompressed_size is None
                       else uncompressed_size)


def gzip_data(data):
    return gzip.compress(data, compresslevel=9, mtime=0)


def page_count_above_chunk():
    # The column chunk declares four values; the page header declares 2^31 - 1.
    column = Column("v", INT64, num_values=ROWS)
    return column.page(DATA_PAGE, INT64_VALUES, num_values=I32_MAX, encoding=PLAIN)


def chunk_count_below_pages():
    # A lax writer: the column chunk declares two values, and its page holds
    # and declares the four values of the four rows.
    column = Column("v", INT64, num_values=2)
    return column.page(DATA_PAGE, INT64_VALUES, num_values=ROWS, encoding=PLAIN)


NULL_ROWS = 100_000


def fixed_len_all_null():
    # Definition levels: a four-byte length, then one RLE run of zeros.
    run = uvarint(NULL_ROWS << 1) + bytes([0])
    levels = struct.pack("<I", len(run)) + run
    column = Column("v", FIXED_LEN_BYTE_ARRAY, repetition=OPTIONAL, type_length=I32_MAX,
                    num_values=NULL_ROWS)
    return column.page(DATA_PAGE, levels, num_values=NULL_ROWS, encoding=PLAIN)


ALLOCATION_BOUNDS = {
    "valid_delta_length": delta_length(),
    "delta_length_count_2e40": delta_length(HUGE),
    "delta_length_zero_width_count_i32_max": delta_length_zero_width(I32_MAX),
    "delta_length_mini_blocks_2e56": delta_length_huge_header(),
    "delta_byte_array_mini_blocks_2e56": delta_byte_array_huge_header(),
    "valid_delta_byte_array": delta_byte_array(),
    "delta_byte_array_prefix_count_2e40": delta_byte_array(prefix_count=HUGE),
    "delta_byte_array_suffix_count_2e40": delta_byte_array(suffix_count=HUGE),
    "valid_dictionary_strings": dictionary_strings(),
    "dictionary_strings_count_i32_max": dictionary_strings(I32_MAX),
    "valid_dictionary_int96": dictionary_int96(),
    "dictionary_int96_count_i32_max": dictionary_int96(I32_MAX),
    "data_page_count_above_chunk": page_count_above_chunk(),
    "valid_chunk_count_below_pages": chunk_count_below_pages(),
    "valid_snappy": compressed(SNAPPY, snappy_literal(INT64_VALUES)),
    "snappy_size_i32_max": compressed(SNAPPY, snappy_literal(INT64_VALUES, I32_MAX), I32_MAX),
    "valid_lz4_raw": compressed(LZ4_RAW, lz4_literal(INT64_VALUES)),
    "lz4_raw_size_i32_max": compressed(LZ4_RAW, lz4_literal(INT64_VALUES), I32_MAX),
    "valid_gzip": compressed(GZIP, gzip_data(INT64_VALUES)),
    # 32 MiB of zeros in 32 KB; the page header declares the 32 bytes of four values.
    "gzip_longer_than_declared": compressed(GZIP, gzip_data(bytes(32 << 20))),
}

# The column chunk and the row group also declare 2^31 - 1 values, so the
# page count is consistent with the footer: only the page bytes contradict it.
LARGE_COUNTS = {
    "delta_length_zero_width_chunk_count_i32_max": delta_length_zero_width(I32_MAX, I32_MAX),
    "delta_byte_array_zero_width_chunk_count_i32_max": delta_byte_array_zero_width(
        I32_MAX, I32_MAX),
}


def write(directory, name, data):
    directory = pathlib.Path(__file__).parent / directory
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.parquet").write_bytes(data)
    print(f"{directory.name}/{name}.parquet: {len(data)} bytes")


if __name__ == "__main__":
    for name, column in ALLOCATION_BOUNDS.items():
        data = parquet_file([int64_column("e", range(ROWS)), column], ROWS)
        write("allocation_bounds", name, data)
    for name, column in LARGE_COUNTS.items():
        data = parquet_file([int64_column("e", range(ROWS)), column], I32_MAX)
        write("allocation_bounds", name, data)
    # A valid file with one column: 100,000 NULL values of a type that is
    # 2^31 - 1 bytes long.
    write("type_length", "fixed_len_type_length_i32_max",
          parquet_file([fixed_len_all_null()], NULL_ROWS))
