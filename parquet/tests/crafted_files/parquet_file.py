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

"""Writes small Parquet files byte by byte, with the Python standard library.

The regression files of this directory need page headers and value encodings
that no Parquet writer produces: a count that the page cannot hold, a checksum
that does not match. This module encodes the Thrift compact protocol, the
value encodings and the file layout directly, so each field of a page header
can have any value.
"""

import struct as _struct
import zlib

# Thrift compact protocol type codes.
TRUE, FALSE, I32, I64, BINARY, LIST, STRUCT = 1, 2, 5, 6, 8, 9, 12

# parquet.thrift enumerations.
BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY = range(8)
REQUIRED, OPTIONAL = 0, 1
UTF8 = 0
PLAIN, RLE, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, RLE_DICTIONARY = (
    0, 3, 5, 6, 7, 8)
UNCOMPRESSED, SNAPPY, GZIP, LZ4_RAW = 0, 1, 2, 7
DATA_PAGE, DICTIONARY_PAGE = 0, 2


def uvarint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def zigzag(value):
    return uvarint((value << 1) if value >= 0 else ((-value) << 1) - 1)


def binary(data):
    return uvarint(len(data)) + data


def thrift_list(element_type, elements):
    if len(elements) < 15:
        header = bytes([(len(elements) << 4) | element_type])
    else:
        header = bytes([0xF0 | element_type]) + uvarint(len(elements))
    return header + b"".join(elements)


def thrift_struct(*fields):
    """Each field is (id, type, encoded value); a field with value None is absent."""
    out = bytearray()
    last = 0
    for field_id, field_type, value in fields:
        if value is None:
            continue
        delta = field_id - last
        if 0 < delta <= 15:
            out.append((delta << 4) | field_type)
        else:
            out.append(field_type)
            out += zigzag(field_id)
        out += value
        last = field_id
    out.append(0)
    return bytes(out)


def i32(field_id, value):
    return (field_id, I32, None if value is None else zigzag(value))


def i64(field_id, value):
    return (field_id, I64, None if value is None else zigzag(value))


def delta_binary_packed(values, declared_count=None):
    """DELTA_BINARY_PACKED data for `values`, in one block of 128 values and
    four mini blocks. `declared_count` replaces the value count of the header."""
    assert 1 <= len(values) <= 33, "one mini block is enough for the regression files"
    count = len(values) if declared_count is None else declared_count
    out = bytearray(uvarint(128) + uvarint(4) + uvarint(count) + zigzag(values[0]))
    deltas = [b - a for a, b in zip(values, values[1:])]
    if deltas:
        min_delta = min(deltas)
        adjusted = [delta - min_delta for delta in deltas]
        width = max(adjusted).bit_length()
        out += zigzag(min_delta) + bytes([width, 0, 0, 0])
        packed = 0
        for index, value in enumerate(adjusted):
            packed |= value << (index * width)
        out += packed.to_bytes(32 * width // 8, "little")
    return bytes(out)


def plain_byte_arrays(values):
    return b"".join(_struct.pack("<I", len(value)) + value for value in values)


def snappy_literal(data, declared_length=None):
    """A Snappy stream with one literal element. `declared_length` replaces the
    uncompressed length at the start of the stream."""
    assert 1 <= len(data) <= 60
    length = len(data) if declared_length is None else declared_length
    return uvarint(length) + bytes([(len(data) - 1) << 2]) + data


def lz4_literal(data):
    """An LZ4 block with one sequence of literals."""
    assert 15 <= len(data) < 15 + 255
    return bytes([0xF0, len(data) - 15]) + data


class Column:
    """One column chunk: a schema element and its pages."""

    def __init__(self, name, physical_type, *, repetition=REQUIRED, converted_type=None,
                 type_length=None, codec=UNCOMPRESSED, num_values):
        self.name = name
        self.physical_type = physical_type
        self.repetition = repetition
        self.converted_type = converted_type
        self.type_length = type_length
        self.codec = codec
        self.num_values = num_values
        self.pages = []  # (header bytes, data bytes, uncompressed size, page type)
        self.encodings = set()

    def page(self, page_type, data, *, num_values, encoding, uncompressed_size=None,
             with_crc=False, crc=None):
        """Adds a page. `data` is the page as stored. `uncompressed_size` and
        `crc` replace the values that are correct for `data`."""
        if uncompressed_size is None:
            uncompressed_size = len(data)
        if with_crc and crc is None:
            crc = zlib.crc32(data)
        if crc is not None and crc >= 1 << 31:
            crc -= 1 << 32
        if page_type == DICTIONARY_PAGE:
            specific = (7, STRUCT, thrift_struct(i32(1, num_values), i32(2, encoding)))
        else:
            specific = (5, STRUCT, thrift_struct(
                i32(1, num_values), i32(2, encoding), i32(3, RLE), i32(4, RLE)))
        header = thrift_struct(
            i32(1, page_type), i32(2, uncompressed_size), i32(3, len(data)), i32(4, crc),
            specific)
        self.pages.append((header, data, uncompressed_size, page_type))
        self.encodings.add(encoding)
        return self

    def schema_element(self):
        return thrift_struct(
            i32(1, self.physical_type), i32(2, self.type_length), i32(3, self.repetition),
            (4, BINARY, binary(self.name.encode())), i32(6, self.converted_type))


def int64_column(name, values, **page_arguments):
    """A valid required INT64 column with PLAIN values."""
    data = b"".join(_struct.pack("<q", value) for value in values)
    column = Column(name, INT64, num_values=len(values))
    return column.page(DATA_PAGE, data, num_values=len(values), encoding=PLAIN,
                       **page_arguments)


def parquet_file(columns, num_rows):
    """The bytes of a file with one row group."""
    out = bytearray(b"PAR1")
    chunks = []
    total_uncompressed = 0
    for column in columns:
        start = len(out)
        dictionary_offset = data_offset = None
        uncompressed = 0
        for header, data, uncompressed_size, page_type in column.pages:
            if page_type == DICTIONARY_PAGE:
                dictionary_offset = len(out)
            elif data_offset is None:
                data_offset = len(out)
            out += header + data
            uncompressed += len(header) + uncompressed_size
        total_uncompressed += uncompressed
        metadata = thrift_struct(
            i32(1, column.physical_type),
            (2, LIST, thrift_list(I32, [zigzag(e) for e in sorted(column.encodings | {RLE})])),
            (3, LIST, thrift_list(BINARY, [binary(column.name.encode())])),
            i32(4, column.codec),
            i64(5, column.num_values),
            i64(6, uncompressed),
            i64(7, len(out) - start),
            i64(9, data_offset),
            i64(11, dictionary_offset))
        chunks.append(thrift_struct(i64(2, 0), (3, STRUCT, metadata)))
    row_group = thrift_struct(
        (1, LIST, thrift_list(STRUCT, chunks)), i64(2, total_uncompressed), i64(3, num_rows))
    root = thrift_struct((4, BINARY, binary(b"schema")), i32(5, len(columns)))
    footer = thrift_struct(
        i32(1, 1),
        (2, LIST, thrift_list(STRUCT, [root] + [c.schema_element() for c in columns])),
        i64(3, num_rows),
        (4, LIST, thrift_list(STRUCT, [row_group])),
        (6, BINARY, binary(b"Longbow crafted_files/parquet_file.py")))
    out += footer + _struct.pack("<I", len(footer)) + b"PAR1"
    return bytes(out)
