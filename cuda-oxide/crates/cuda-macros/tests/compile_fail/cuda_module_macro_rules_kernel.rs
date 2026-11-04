// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// An attribute macro receives its input before inner macro invocations are
// expanded, so `#[cuda_module]` cannot discover a kernel that a local
// `macro_rules!` generates. The diagnostic must name that boundary and point
// at the generate-the-whole-module workaround.

use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    macro_rules! fill_kernel {
        ($name:ident) => {
            #[cuda_macros::kernel]
            pub fn $name(value: u32) {
                let _ = value;
            }
        };
    }

    fill_kernel!(fill);
}

fn main() {}
