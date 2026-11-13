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

"""Writes the regression files of `checksums/`.

A page header can carry a CRC-32 of the page as stored. The valid files have
correct checksums, or none. In each invalid file one page and its checksum
disagree: without verification the file reads without an error and returns
other values. Run this script in its directory to write the files again; the
output is deterministic.
"""

import pathlib
import struct
import zlib

from parquet_file import (
    BYTE_ARRAY, DATA_PAGE, DICTIONARY_PAGE, INT64, PLAIN, RLE_DICTIONARY, SNAPPY, UTF8, Column,
    int64_column, parquet_file, plain_byte_arrays, snappy_literal)

ROWS = 4
# RLE_DICTIONARY indices 0, 1, 0, 1: bit width 1, one bit-packed group.
INDICES = bytes([1, 0x03, 0b1010])


def int64_values(values):
    return b"".join(struct.pack("<q", value) for value in values)


def strings(with_crc, dictionary=(b"x", b"yy"), dictionary_crc=None):
    column = Column("v", BYTE_ARRAY, converted_type=UTF8, num_values=ROWS)
    column.page(DICTIONARY_PAGE, plain_byte_arrays(dictionary), num_values=2, encoding=PLAIN,
                with_crc=with_crc, crc=dictionary_crc)
    return column.page(DATA_PAGE, INDICES, num_values=ROWS, encoding=RLE_DICTIONARY,
                       with_crc=with_crc)


def snappy(values, crc=None):
    data = snappy_literal(int64_values(values))
    column = Column("s", INT64, codec=SNAPPY, num_values=ROWS)
    return column.page(DATA_PAGE, data, num_values=ROWS, encoding=PLAIN, uncompressed_size=32,
                       with_crc=True, crc=crc)


def columns(with_crc=True, e=range(ROWS), e_crc=None, dictionary=(b"x", b"yy"),
            dictionary_crc=None, s=range(10, 10 + ROWS), s_crc=None):
    return [
        int64_column("e", e, with_crc=with_crc, crc=e_crc),
        strings(with_crc, dictionary, dictionary_crc),
        # The checksum of a compressed page is of the compressed bytes.
        snappy(s, s_crc) if with_crc else snappy_without_crc(s),
    ]


def snappy_without_crc(values):
    column = Column("s", INT64, codec=SNAPPY, num_values=ROWS)
    return column.page(DATA_PAGE, snappy_literal(int64_values(values)), num_values=ROWS,
                       encoding=PLAIN, uncompressed_size=32)


VALID_E = zlib.crc32(int64_values(range(ROWS)))
VALID_DICTIONARY = zlib.crc32(plain_byte_arrays((b"x", b"yy")))
VALID_S = zlib.crc32(snappy_literal(int64_values(range(10, 10 + ROWS))))

FILES = {
    "valid_with_checksums": columns(),
    "valid_without_checksums": columns(with_crc=False),
    # Column e: the value of the third row is 258, the checksum is of the value 2.
    "data_page_byte_changed": columns(e=[0, 1, 258, 3], e_crc=VALID_E),
    # Column v: the first dictionary value is "z", the checksum is of "x".
    "dictionary_page_byte_changed": columns(dictionary=(b"z", b"yy"),
                                            dictionary_crc=VALID_DICTIONARY),
    # Column s: the compressed page holds 10, 11, 12, 77; the checksum is of 10 to 13.
    "compressed_page_byte_changed": columns(s=[10, 11, 12, 77], s_crc=VALID_S),
    # Column e: the page is unchanged; one bit of the checksum is changed.
    "checksum_changed": columns(e_crc=VALID_E ^ 1),
}

if __name__ == "__main__":
    directory = pathlib.Path(__file__).parent / "checksums"
    directory.mkdir(exist_ok=True)
    for name, file_columns in FILES.items():
        data = parquet_file(file_columns, ROWS)
        (directory / f"{name}.parquet").write_bytes(data)
        print(f"checksums/{name}.parquet: {len(data)} bytes")
