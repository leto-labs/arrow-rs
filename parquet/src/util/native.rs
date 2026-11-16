// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Exact primitive layout contract shared by native dictionary and NULL expansion.
use std::any::TypeId;

#[inline]
pub(crate) fn native<T: 'static>() -> bool {
    let t = TypeId::of::<T>();
    t == TypeId::of::<i32>()
        || t == TypeId::of::<i64>()
        || t == TypeId::of::<f32>()
        || t == TypeId::of::<f64>()
}

/// Per-thread test controls never bypass CPU feature detection. A scalar tier
/// disables POPCNT specialization too; AVX2 uses portable NULL expansion.
///
/// `PARQUET_TEST_NATIVE_TIER` (`scalar`, `avx2`, `avx512` or `auto`, the
/// default) sets the starting tier of every test thread, so a whole test run
/// can be forced to one tier. [`for_each_tier`] then runs only that tier and
/// fails if this CPU does not support it.
#[cfg(test)]
pub(crate) mod dispatch {
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Tier {
        Auto,
        Scalar,
        Avx2,
        Avx512,
    }
    thread_local! { static TIER: Cell<Tier> = Cell::new(forced()); }

    /// The tier named by `PARQUET_TEST_NATIVE_TIER`, or `Auto`.
    fn forced() -> Tier {
        match std::env::var("PARQUET_TEST_NATIVE_TIER").as_deref() {
            Err(_) | Ok("" | "auto") => Tier::Auto,
            Ok("scalar") => Tier::Scalar,
            Ok("avx2") => Tier::Avx2,
            Ok("avx512") => Tier::Avx512,
            Ok(other) => panic!("unknown PARQUET_TEST_NATIVE_TIER {other:?}"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn tier() -> Tier {
        TIER.get()
    }

    struct Restore(Tier);
    impl Drop for Restore {
        fn drop(&mut self) {
            TIER.set(self.0);
        }
    }

    pub(crate) fn for_each_tier(mut property: impl FnMut()) {
        let forced = forced();
        for tier in [Tier::Scalar, Tier::Avx2, Tier::Avx512] {
            if forced != Tier::Auto && tier != forced {
                continue;
            }
            let supported = tier == Tier::Scalar;
            #[cfg(target_arch = "x86_64")]
            let supported = supported
                || (!cfg!(miri)
                    && match tier {
                        Tier::Avx2 => std::is_x86_feature_detected!("avx2"),
                        Tier::Avx512 => {
                            std::is_x86_feature_detected!("avx512f")
                                && std::is_x86_feature_detected!("avx512bw")
                                && std::is_x86_feature_detected!("avx512vl")
                                && std::is_x86_feature_detected!("avx512vbmi")
                                && std::is_x86_feature_detected!("popcnt")
                        }
                        _ => false,
                    });
            if !supported {
                assert!(
                    forced == Tier::Auto,
                    "PARQUET_TEST_NATIVE_TIER={tier:?} is not supported by this CPU"
                );
                continue;
            }
            let _restore = Restore(TIER.replace(tier));
            eprintln!("native property tier: {tier:?}");
            property();
        }
    }
}
