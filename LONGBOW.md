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
