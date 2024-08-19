# mypy: allow-untyped-defs
import contextlib
import logging
import math
from functools import lru_cache
from typing import Any, Callable, cast, List, Optional, Set, Union
from unittest.mock import patch

import torch
import torch.utils

from ..._dynamo.utils import counters
from .. import config, ir, lowering as L
from ..kernel.mm_common import mm_args, is_woq_gemm
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import cache_on_self, has_free_symbols, parallel_num_threads
from ..virtualized import V
from .cpp import get_export_declaration
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm, LayoutType
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}


{%- if qGroupSize is not none %}
const bfloat16* dequant(
    const int* in_ptr,
    const bfloat16* ScaleAndZeros,
    const long* qGroupSize,
    bfloat16* out_ptr,
    long block_n_size,
    long block_k_size,
    long ldb,
    long bn,
    long k_start){

    long N = bn * block_n_size;

    static constexpr float lut[16] = {
        -8.0f, -7.0f, -6.0f, -5.0f,
        -4.0f, -3.0f, -2.0f, -1.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };

    const unsigned char* in_ptr_cast = reinterpret_cast<const unsigned char*>(in_ptr);


#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
    {{kernel.assert_function}}(
        block_n_size % 16 == 0,
        "Not all partitions are assigned."
    );
    using Vectorized_fp32 = at::vec::Vectorized<float>;
    using Vectorized_bf16 = at::vec::Vectorized<bfloat16>;
    for (int k = 0; k < block_k_size; k += 2) {
        for (int n = 0; n < block_n_size; n += 16) {
            int kb = (k_start + k) / qGroupSize[0] - k_start / qGroupSize[0];
            int kb2 = (k_start + k + 1) / qGroupSize[0] - k_start / qGroupSize[0];

            // TODO: Support the block_k_size cross the group_size
            // if (kb == kb2) {
            // } else 

            // Step 1: Get the scale and zero point         
            // s0, z0, s2, z2 .... s30, z30
            const auto scale_zp = Vectorized_bf16::loadu(ScaleAndZeros + (kb * N * 2 + n * 2));
            // s1, z1, s3 z3 .... s31, z31
            const auto scale_zp2 = Vectorized_bf16::loadu(ScaleAndZeros + (kb2 * N * 2 + n * 2));

            // s0, s2, ... s30
            __m512 scale1, scale2, zp1, zp2;
            at::vec::cvtbf16_fp32(
                _mm512_cvtepi32_epi16(scale_zp),
                scale1
            );
            // s1, s3, ... s31
            at::vec::cvtbf16_fp32(
                _mm512_cvtepi32_epi16(scale_zp2),
                scale2
            );
            // zp0, zp2, ... zp30
            at::vec::cvtbf16_fp32(
                _mm512_cvtepi32_epi16(
                    _mm512_srli_epi32(scale_zp, 16)
                ),
                zp1
            );
            // zp1, zp3, ... zp31
            at::vec::cvtbf16_fp32(
                _mm512_cvtepi32_epi16(
                    _mm512_srli_epi32(scale_zp2, 16)
                ),
                zp2
            );
                                                         
            // Step 2: load the wgt data
            // total 32 elements with 4x2 * 16
            __m128i b4 = _mm_loadu_si128((__m128i*)(in_ptr_cast + (k / 2 * block_n_size + n)));
            const __m128i lowMask = _mm_set1_epi8(0xF);
            __m128i high = _mm_andnot_si128(lowMask, b4);
            __m128i low = _mm_and_si128(lowMask, b4);

            __m512i high_512 = _mm512_cvtepu16_epi32(
                _mm256_srli_epi16(_mm256_cvtepu8_epi16(high), 4)
            );
            __m512i low_512 = _mm512_cvtepu16_epi32(
                _mm256_cvtepu8_epi16(low)
            );

            static const __m512 lut = _mm512_set_ps(
                7.0f, 6.0f, 5.0f, 4.0f,
                3.0f, 2.0f, 1.0f, 0.0f,
                -1.0f, -2.0f, -3.0f, -4.0f,
                -5.0f, -6.0f, -7.0f, -8.0f);

            // Step 3: mul scale and add zp
            const __m512 b_val = _mm512_permutexvar_ps(high_512, lut) * scale1 + zp1;
            const __m512 b_val2 = _mm512_permutexvar_ps(low_512, lut) * scale2 + zp2;

            // cvt 2 FP32 to 1 BF16
            const __m512i b_val_bf16 = at::vec::cvtfp32_bf16(b_val2, b_val);

            // Step 4: shuffle the BF16 by interleave
            //  From: b0, b1, ... b15, b16, b17, ... b31
            //  TO: b0, b16, b1, b17, b2, b18, ... b15, b31
            static const uint16_t control[16] = {
                0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
            };
            __m256i vec_control = _mm256_loadu_si256((__m256i*)control);

            static const uint16_t control2[16] = {
                8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
            };
            __m256i vec_control2 = _mm256_loadu_si256((__m256i*)control2);
                                                
            __m256i lower_256 = _mm512_castsi512_si256(b_val_bf16);
            __m256i higher_256 = _mm512_extracti64x4_epi64(b_val_bf16, 1);

            __m256i shuffled = _mm256_permutex2var_epi16(lower_256, vec_control, higher_256);
            __m256i shuffled2 = _mm256_permutex2var_epi16(lower_256, vec_control2, higher_256);
            
            __m512i vec_512 = _mm512_castsi256_si512(shuffled2);
            vec_512 = _mm512_inserti64x4(vec_512, shuffled, 1);

            // Step 5: store the result back
            {%- if micro_gemm.get_b_layout().value == 1 %}
            // VNNI2 layout
            // out_ptr[k / 2 * block_n_size * 2 + n * 2 + k % 2] = static_cast<bfloat16>(b_val);
            _mm512_storeu_si512((__m512i*)(out_ptr + (k / 2 * block_n_size * 2 + n * 2)), vec_512);
            {%- else %}
            // For ref micro gemm, write to contiguous layout
            // out_ptr[k * block_n_size + n] = static_cast<bfloat16>(b_val);
            {%- endif %}
        }
    }
#else
    for (int k = 0; k < block_k_size; k += 1) {
        for (int n = 0; n < block_n_size; n += 1) {
            int k_out = k_start + k;
            int kb = k_out / qGroupSize[0] - k_start / qGroupSize[0];
                                                                       
            const auto scale = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2]);
            const auto zero = static_cast<float>(ScaleAndZeros[kb * N * 2 + n * 2 + 1]);
                                                                       
            long idx = (k / 2 * block_n_size + n);
            long offset = 1 - k % 2;
            unsigned char val = in_ptr_cast[idx];
            int index = ((val & (0xF << (offset * 4))) >> (offset * 4));
            const float b_val = lut[index] * scale + zero;

            // Important: need to write output_ptr with vnni2 layout
            {%- if micro_gemm.get_b_layout().value == 1 %}
            // VNNI2 layout
            out_ptr[k / 2 * block_n_size * 2 + n * 2 + k % 2] = static_cast<bfloat16>(b_val);
            {%- else %}
            // For ref micro gemm, write to contiguous layout
            out_ptr[k * block_n_size + n] = static_cast<bfloat16>(b_val);
            {%- endif %}
            // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" scale is: "<<scale<<" zero is: "<<zero<<std::endl;
            // std::cout<<"---- n is: "<<n<<" k is: "<<k<<" index is: "<<index<<" val is: "<<static_cast<int>(val)<<" deqaunt_val is: "<<b_val<<std::endl;
        }
    }
#endif
    return out_ptr;
}
{%- endif %}


{{micro_gemm.codegen_define(kernel)}}

{%- if x_scale is not none %}
{%- set kernel_args = {"X": X, "W": W, "inp": inp, "x_scale": x_scale, "x_zp": x_zp, "w_scale": w_scale, "w_zp": w_zp,} %}
{%- else %}

{%- if qGroupSize is not none %}
{%- set kernel_args = {"X": X, "W": W, "qGroupSize": qGroupSize, "ScaleZP": ScaleZP} %}
{%- else %}
{%- set kernel_args = {"X": X, "W": W, "inp": inp} %}
{%- endif %}


{%- endif %}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs={"Y": Y}, aliases=aliases)}}
{
    {{kernel.maybe_codegen_profile()}}
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{N}};
    constexpr int64_t K = {{K}};
    constexpr int64_t M0 = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t N0 = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t K0 = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    {%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, 0)}};
    const int64_t M0_blocks = (M + M0 - 1) / M0;
    {%- if num_threads > 1 %}
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, M, N, K, M0, N0, K0, Mt_blocks, Nt_blocks, Kt_blocks);
    {%- else %}
    const auto Mt_blocks = M0_blocks;
    const auto Nt_blocks = N0_blocks;
    const auto Kt_blocks = K0_blocks;
    {%- endif %}
    const int64_t Mc_blocks = Mt_blocks;
    const int64_t Kc_blocks = Kt_blocks;
    const int64_t num_Mc_blocks = (M0_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks = N0_blocks;
    const int64_t num_k_slices = (K0_blocks + Kt_blocks - 1) / Kt_blocks;
    {%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, 0)}};
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = {{template.thread_blocking().block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking().block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking().block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking().block_m}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking().block_k}};
    constexpr int64_t num_Mc_blocks = (M0_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = N0_blocks;
    constexpr int64_t num_k_slices = (K0_blocks + Kt_blocks - 1) / Kt_blocks;
    {%- endif %}

    // make sure all partitions are assigned
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );

    {%- if maybe_k_slicing %}
    std::unique_ptr<std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[]> local_buf_ptrs;
    if (num_k_slices > 1) {
        local_buf_ptrs.reset(new std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[num_Mc_blocks * num_Nc_blocks * num_k_slices]);
    }
    {%- endif %}

    {%- if num_threads > 1 %}
    #pragma omp parallel num_threads({{num_threads}})
    {
        const int tid = omp_get_thread_num();
        int64_t m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end;
        mm_get_thread_blocks(
            tid, M0_blocks, N0_blocks, K0_blocks, Mt_blocks, Nt_blocks, Kt_blocks,
            m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end);
        {%- if maybe_k_slicing %}
        const int64_t k_group_id = tid / num_k_slices;
        const int64_t k_slice_id = tid % num_k_slices;
        {%- endif %}
    {%- else %}
    {
        const int tid = 0;
        const int64_t m_block_start = 0;
        const int64_t m_block_end = M0_blocks;
        const int64_t n_block_start = 0;
        const int64_t n_block_end = N0_blocks;
        const int64_t k_block_start = 0;
        const int64_t k_block_end = K0_blocks;
    {%- endif %}
        {{ micro_gemm.codegen_init(kernel) }}
        for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
            const int64_t m_start = mc * M0;
            const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * M0, M);
            const int64_t m_size = m_end - m_start;
            {%- if use_local_acc %}
            {%- set acc_buf_name = "local_acc_buf" %}
            {{ kernel.define_buffer(acc_buf_name, ["m_end - m_start", "N0"], acc_buf_dtype) }}
            {%- endif %}
            for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                const int64_t n_start = nc * N0;
                const int64_t n_end = std::min((nc + 1) * N0, N);
                const int64_t n_size = n_end - n_start;
                {%- if use_local_acc %}
                {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
                {%- else %}
                {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                {%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * K0;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * K0, K);
                    {%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}

                    {%- if qGroupSize is not none %}
                    int64_t k_start_woq = kc * K0 / 8;
                    int64_t k_end_woq = std::min(std::min(kc + Kc_blocks, k_block_end) * K0, K) / 8;
                    {%- set tile_W_3d = kernel.slice_nd(W, [("nc", "nc + 1"), ("k_start_woq", "k_end_woq"), ()]) %}
                    {%- set tile_W = kernel.view(tile_W_3d, ["k_end_woq - k_start_woq", micro_gemm.register_blocking.block_n]) %}    
                    {%- else %}
                    {%- set tile_W_3d = kernel.slice_nd(W, [("nc", "nc + 1"), ("k_start", "k_end"), ()]) %}
                    {%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}                
                    {%- endif %}
                    
                    {%- if qGroupSize is not none %}

                    {%- set dequant_buf_name = "int4_dequant_buf" %}
                    {{ kernel.define_buffer(dequant_buf_name, ["k_end - k_start", micro_gemm.register_blocking.block_n], dequant_buf_dtype) }}
                    {%- set dequant_buf = kernel.local_buffers[dequant_buf_name] %}
                    int64_t qGroupSize_int = *qGroupSize;
                    int64_t k_start_woq_scale_zp = k_start / qGroupSize_int;
                    int64_t k_end_woq_scale_zp = (k_end + qGroupSize_int - 1) / qGroupSize_int;
                    {%- set ScaleZP_View = kernel.slice_nd(ScaleZP, [("k_start_woq_scale_zp", "k_end_woq_scale_zp"), ("n_start", "n_end"), ()]) %}
                    {{ kernel.dequant(tile_W, ScaleZP_View, dequant_buf, X, W, qGroupSize, "k_start") }}
                    {%- set tile_W = dequant_buf %}
                    {%- endif %}

                    if (kc == k_block_start) {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=False)|indent(24, false) }}
                    } else {
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True)|indent(24, false) }}
                    }

                }
                {%- if maybe_k_slicing %}
                if (num_k_slices > 1) {
                    const int64_t mxn_cache_block_id = mc * num_Nc_blocks + nc;
                    local_buf_ptrs[mxn_cache_block_id * num_k_slices + k_slice_id].reset({{ kernel.release_buffer(acc_buf_name) }});
                } else
                {%- endif %}
                {
                {%- if N == PADDED_N %}
                    {%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                    {%- set tile_acc = acc %}
                {%- else %}
                    {%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
                    {%- set tile_acc = kernel.slice_nd(acc, [(), ("0", "n_end - n_start")]) %}
                {%- endif %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
        {%- if maybe_k_slicing %}
        if (num_k_slices > 1) {
            #pragma omp barrier
            for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
                // We slice M-dim and each thread in the k-slicing group works on a slice
                const int64_t m_start_unsliced = mc * M0;
                const int64_t m_end_unsliced = std::min(std::min(mc + Mc_blocks, m_block_end) * M0, M);
                const int64_t m_size_unsliced = m_end_unsliced - m_start_unsliced;
                const int64_t m_slice_size = (m_size_unsliced + num_k_slices - 1) / num_k_slices;
                const int64_t m_start = std::min(m_start_unsliced + m_slice_size * k_slice_id, m_end_unsliced);
                const int64_t m_end = std::min(m_start_unsliced + m_slice_size * (k_slice_id + 1), m_end_unsliced);
                const int64_t m_size = m_end - m_start;
                const int64_t m_offset = m_start - m_start_unsliced;
                for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                    const int64_t n_start = nc * N0;
                    const int64_t n_end = std::min((nc + 1) * N0, N);
                    const int64_t n_size = n_end - n_start;
                    const int64_t mxn_cache_block_id = mc * num_Nc_blocks + nc;
                    auto {{acc_buf_name}} = local_buf_ptrs[mxn_cache_block_id * num_k_slices].get();
                    for (int64_t other_slice = 1; other_slice < num_k_slices; other_slice++) {
                        auto other_acc = local_buf_ptrs[mxn_cache_block_id * num_k_slices + other_slice].get();
                        for (int64_t m = m_offset; m < m_offset + m_size; m++) {
                            #pragma omp simd
                            for (int64_t n = 0; n < n_size; n++) {
                                {{acc_buf_name}}[m*N0 + n] += other_acc[m*N0 + n];
                            }
                        }
                    }
                    {%- set tile_acc_m_slice = kernel.slice_nd(tile_acc, [("m_offset", "m_offset + m_end - m_start"), ()]) %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc_m_slice, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
        {%- endif %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
}
"""


def get_padded_n(n, block_n):
    return (n + block_n - 1) // block_n * block_n


class CppPackedGemmTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16, torch.half, torch.uint8]
        super().__init__(
            "packed_gemm",
            input_nodes,
            layout,
            num_threads,
            epilogue_creator=epilogue_creator,
        )
        self.beta = beta
        self.alpha = alpha
        self.has_bias = has_bias
        self.register_blocking = register_blocking
        m, n = layout.size
        _, k = input_nodes[0].get_size()
        self.m, self.n, self.k = m, n, k
        self.padded_n = get_padded_n(n, self.register_blocking.block_n)
        self.is_dynamic_M = has_free_symbols((m,))

    @cache_on_self
    def thread_blocking(self) -> GemmBlocking:
        """
        NOTE [Thread blocking in Cpp GEMM]
        We use simple heuristics to decide the thread blocking:
        1. Make sure all threads are occupied as much as possible.
        2. For (m, n) blocks, favor more square-sized thread blocks for better data reuse.
        3. If (m, n) blocks cannot occupy all the threads, we consider k-slicing.
        TODO(jgong5): allow tuning various blocking options
        """

        @lru_cache(maxsize=100)
        def get_factors(number):
            factors = []
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(m_factor, n_factor, k_factor, m_blocks, n_blocks, k_blocks):
            thread_block_k = math.ceil(k_blocks / k_factor)
            thread_block_n = math.ceil(n_blocks / n_factor)
            thread_block_m = math.ceil(m_blocks / m_factor)
            return GemmBlocking(thread_block_m, thread_block_n, thread_block_k)

        assert (
            not self.is_dynamic_M
        ), "Unable to determine thread blocking for dynamic M."
        register_blocking = self.register_blocking
        m_blocks = math.ceil(self.m / register_blocking.block_m)
        n_blocks = math.ceil(self.n / register_blocking.block_n)
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        factors = get_factors(self.num_threads)
        assert len(factors) > 0

        # we favor square-sized thread blocks for good data reuse
        def get_better_blocking(blocking, best_blocking):
            if best_blocking is None:
                best_blocking = blocking
            else:
                block_m_size = blocking.block_m * register_blocking.block_m
                block_n_size = blocking.block_n * register_blocking.block_n
                best_block_m_size = best_blocking.block_m * register_blocking.block_m
                best_block_n_size = best_blocking.block_n * register_blocking.block_n
                if blocking.block_k > best_blocking.block_k:
                    best_blocking = blocking
                elif (
                    blocking.block_k == best_blocking.block_k
                    and block_m_size + block_n_size
                    < best_block_m_size + best_block_n_size
                ):
                    best_blocking = blocking
            return best_blocking

        best_blocking = None
        # check if we can have a thread-blocking to occupy all threads without k-slicing
        for n_factor in factors:
            m_factor = self.num_threads // n_factor
            if n_blocks >= n_factor and m_blocks >= m_factor:
                blocking = get_blocking(
                    m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                )
                best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for k_factor in factors:
                if k_blocks >= k_factor and (
                    config.cpp.gemm_max_k_slices == 0
                    or k_factor <= config.cpp.gemm_max_k_slices
                ):
                    n_factors = get_factors(self.num_threads // k_factor)
                    for n_factor in n_factors:
                        m_factor = (self.num_threads // k_factor) // n_factor
                        if n_blocks >= n_factor and m_blocks >= m_factor:
                            blocking = get_blocking(
                                m_factor,
                                n_factor,
                                k_factor,
                                m_blocks,
                                n_blocks,
                                k_blocks,
                            )
                            best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for n_factor in factors:
                m_factor = self.num_threads // n_factor
                if n_blocks >= n_factor or m_blocks >= m_factor:
                    blocking = get_blocking(
                        m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                    )
                    best_blocking = get_better_blocking(blocking, best_blocking)

        assert best_blocking is not None
        return best_blocking

    @cache_on_self
    def cache_blocking(self) -> GemmBlocking:
        def get_cache_blocking(register_blocking, thread_blocking):
            M0 = register_blocking.block_m
            N0 = register_blocking.block_n
            K0 = register_blocking.block_k

            Mc_blocks = thread_blocking.block_m
            # Nc_blocks is always 1
            Nc_blocks = 1
            Kc_blocks = thread_blocking.block_k

            # TODO: tune the factor here
            L1_limit_factor = 1
            L2_limit_factor = 0.5

            L1_cache_size = (
                torch._C._cpu._L1d_cache_size()
            )  # per core cache size in Bytes
            assert (
                L1_cache_size > 0
            ), f"Expect L1_cache_size > 0 but got {L1_cache_size}"
            L2_cache_size = (
                torch._C._cpu._L2_cache_size()
            )  # per core cache size in Bytes
            assert (
                L2_cache_size > 0
            ), f"Expect L2_cache_size > 0 but got {L2_cache_size}"
            B_size_limit = L1_cache_size * L1_limit_factor
            A_size_limit = L2_cache_size * L2_limit_factor

            def get_num_byte(dtype):
                return torch.tensor([], dtype=dtype).element_size()

            num_byte_A = get_num_byte(self.input_nodes[0].get_dtype())
            num_byte_B = get_num_byte(self.input_nodes[1].get_dtype())

            size_cache_B = K0 * Kc_blocks * N0 * Nc_blocks * num_byte_B

            if size_cache_B > B_size_limit:
                Kc_blocks = math.floor(
                    B_size_limit / (K0 * N0 * Nc_blocks * num_byte_B)
                )

            size_cache_A = M0 * Mc_blocks * K0 * Kc_blocks * num_byte_A
            if size_cache_A > A_size_limit:
                Mc_blocks = math.floor(
                    A_size_limit / (M0 * Kc_blocks * K0 * num_byte_A)
                )

            return Mc_blocks, Nc_blocks, Kc_blocks

        assert (
            not self.is_dynamic_M
        ), "Unable to determine cache blocking for dynamic M."
        register_blocking = self.register_blocking
        thread_blocking = self.thread_blocking()

        return GemmBlocking(*get_cache_blocking(register_blocking, thread_blocking))

    def log_blockings(self):
        log.debug(f"Register blocking: {self.register_blocking}")  # noqa: G004
        if self.is_dynamic_M:
            # thread and cache blockings are determined at runtime for dynamic shapes
            return
        log.debug(f"Cache blocking: {self.cache_blocking()}")  # noqa: G004
        thread_blocking = self.thread_blocking()
        log.debug(f"Thread blocking: {thread_blocking}")  # noqa: G004

        def get_occupancy():
            m_blocks = math.ceil(self.m / self.register_blocking.block_m)
            n_blocks = math.ceil(self.n / self.register_blocking.block_n)
            k_blocks = math.ceil(self.k / self.register_blocking.block_k)
            m = math.ceil(m_blocks / thread_blocking.block_m)
            n = math.ceil(n_blocks / thread_blocking.block_n)
            k = math.ceil(k_blocks / thread_blocking.block_k)
            return (m, n, k)

        log.debug(
            f"Number of threads: {self.num_threads}, occupancy: {get_occupancy()}"  # noqa: G004
        )

    def maybe_k_slicing(self):
        if self.num_threads == 1:
            return False
        if self.is_dynamic_M:
            # TODO(jgong5): perhaps use size hint to decide?
            return True
        register_blocking = self.register_blocking
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        thread_blocking = self.thread_blocking()
        return k_blocks > thread_blocking.block_k

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        if input_indices is None:
            input_indices = list(range(len(input_nodes)))

        def reorder_and_filter(inputs, layout_or_out):
            if has_bias:
                assert len(input_indices) >= 3
                # Assume the input order is [inp, x, w] and we reorder it to [x, w, inp]
                inp_idx = input_indices[0]
                x_idx = input_indices[1]
                w_idx = input_indices[2]
                return [
                    inputs[x_idx],
                    inputs[w_idx],
                    inputs[inp_idx],
                    *[inputs[idx] for idx in input_indices[3:]],
                ], layout_or_out
            else:
                assert len(input_indices) >= 2
                return [inputs[idx] for idx in input_indices], layout_or_out

        def maybe_to_dense(inputs, layout_or_out):
            new_inputs = list(inputs)
            if isinstance(inputs[1], torch.Tensor):
                W = inputs[1]
                new_inputs[1] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(inputs, layout_or_out):
            woq_gemm = is_woq_gemm(inputs[0], inputs[1])

            if not trans_w:
                return inputs, layout_or_out
            new_inputs = list(inputs)
            X = inputs[0]
            W = inputs[1]
            B = inputs[2] if (len(inputs) > 2 and not woq_gemm) else None
            if isinstance(W, ir.IRNode):
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                if woq_gemm:
                    # W has been packed to 4D of int32
                    # View to [N, (K // 8)] of int32
                    mat2_num_elem = torch.prod(torch.tensor(W.get_size(), dtype=torch.int)).item()
                    W = L.view(W, [mat2_num_elem // X.get_size()[1] * 8, X.get_size()[1] // 8])
                if trans_w:
                    W = L.permute(W, [1, 0])
            else:
                if woq_gemm:
                    # W has been packed to 4D of int32
                    # View to [N, (K // 8)] of int32
                    mat2_num_elem = torch.prod(torch.tensor(W.size(), dtype=torch.int)).item()
                    inner_k_tiles = 8
                    K0 = inputs[0].size()[-1] if isinstance(inputs[0], torch.Tensor) else inputs[0].get_size()[-1]
                    # unpack W: [N, K//2] of uint8 
                    W = torch.ops.aten._convert_weight_to_int4unpack(W, inner_k_tiles, K0)
                    assert trans_w
                    assert isinstance(W, torch.Tensor)
                    # transpose to [K//2, N] of uint8
                    W = W.transpose(0, 1).contiguous()
                    # view back to [K//2, N//4] of int32
                    W = W.view(torch.int32)
                else:
                    if trans_w:
                        assert isinstance(W, torch.Tensor)
                        W = W.transpose(0, 1)
            if B is not None:
                if isinstance(B, ir.IRNode):
                    if not isinstance(B, ir.TensorBox):
                        B = ir.TensorBox(B)
                    B = L.expand(B, (X.get_size()[0], B.get_size()[-1]))
                else:
                    assert isinstance(B, torch.Tensor)
                    B = B.expand(X.shape[0], B.shape[-1])
            new_inputs[1] = W
            if B is not None:
                new_inputs[2] = B
            return new_inputs, layout_or_out

        # TODO(jgong5): decide proper number of threads per problem size
        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(
            *maybe_to_dense(*reorder_and_filter(input_nodes, layout))
        )
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[0].get_dtype() if is_woq_gemm(new_inputs[0], new_inputs[1]) else new_inputs[1].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
        )
        assert micro_gemm is not None
        _, block_n, _ = micro_gemm.register_blocking
        padded_n = get_padded_n(n, block_n)

        def pack_weight(inputs, layout_or_out):
            woq_gemm = is_woq_gemm(inputs[0], inputs[1])

            W = inputs[1]
            new_inputs = list(inputs)
            blocked_w: Union[ir.IRNode, torch.Tensor] = W
            if isinstance(W, ir.IRNode):
                if woq_gemm:
                    # W is with size [K//2, N//4] of int32
                    new_size = [padded_n // block_n, k//8, block_n]
                    blocked_w = ir.Buffer(
                        W.get_name(),  # Borrow the registered buffer name
                        ir.FixedLayout(
                            W.get_device(),
                            W.get_dtype(),
                            new_size,
                            ir.FlexibleLayout.contiguous_strides(new_size),
                            0,
                        ),
                    )
                else:
                    new_size = [padded_n // block_n, k, block_n]
                    blocked_w = ir.Buffer(
                        W.get_name(),  # Borrow the registered buffer name
                        ir.FixedLayout(
                            W.get_device(),
                            W.get_dtype(),
                            new_size,
                            ir.FlexibleLayout.contiguous_strides(new_size),
                            0,
                        ),
                    )
            else:
                if woq_gemm:
                    # W is unpacked to size [K//2, N//4] of int32
                    # view back to [K//2, N] of uint8
                    W = W.view(torch.uint8)

                    # blocked_w: [padded_n // block_n, k//2, block_n] of uint8
                    blocked_w = (
                        torch.nn.functional.pad(W, (0, padded_n - n))
                        .reshape(k//2, padded_n // block_n, block_n)
                        .transpose(0, 1)
                        .contiguous()
                    )

                    # blocked_w view to [padded_n // block_n, k//8, block_n] of int32
                    blocked_w = blocked_w.flatten().view(torch.int32)
                    blocked_w = blocked_w.reshape(padded_n // block_n, k//8, block_n)

                else:
                    blocked_w = (
                        torch.nn.functional.pad(W, (0, padded_n - n))
                        .reshape(k, padded_n // block_n, block_n)
                        .transpose(0, 1)
                        .contiguous()
                    )
                if (
                    micro_gemm.get_b_layout() != LayoutType.NORMAL
                    and not woq_gemm  # BF16-INT4 gemm has 4x2 storage
                ):
                    layout_str = (
                        "VNNI4"
                        if micro_gemm.get_b_layout() == LayoutType.VNNI4
                        else "VNNI2"
                    )
                    assert micro_gemm.get_b_layout() in [
                        LayoutType.VNNI2,
                        LayoutType.VNNI4,
                    ], f"We only support {layout_str} for now"
                    vnni_size = (
                        4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                    )
                    assert (
                        k % vnni_size == 0
                    ), f"k should be divisible by vnni_size for {layout_str} layout"
                    blocked_w = (
                        blocked_w.view(
                            padded_n // block_n, k // vnni_size, vnni_size, block_n
                        )
                        .transpose(-1, -2)
                        .contiguous()
                        .view(padded_n // block_n, k, block_n)
                    )
                # normalize stride to be "contiguous_strides" per size
                # this avoids the problems in L.view during template codegen
                new_stride = [1]
                for sz in reversed(blocked_w.shape[1:]):
                    new_stride.insert(0, new_stride[0] * sz)
                blocked_w = blocked_w.as_strided(blocked_w.shape, new_stride)
            new_inputs[1] = blocked_w

            def _is_int8_gemm(inputs):
                return (
                    isinstance(inputs[0], ir.IRNode)
                    and inputs[0].get_dtype() == torch.uint8
                ) or (
                    isinstance(inputs[0], torch.Tensor)
                    and inputs[0].dtype == torch.uint8
                )

            if _is_int8_gemm(new_inputs):
                BCompensate = None
                if isinstance(W, ir.IRNode):
                    BCompensate = V.graph.add_tensor_constant(
                        V.graph.constants[W.get_name() + "_BMatrixCompens"],
                        W.get_name() + "_BMatrixCompens",
                    )
                else:
                    BCompensate = torch.sum(W.to_dense().to(torch.float), dim=0)  # type: ignore[assignment]
                new_inputs.append(BCompensate)
            return new_inputs, layout_or_out

        def preprocessor(inputs, layout):
            return pack_weight(
                *normalize_shapes(*maybe_to_dense(*reorder_and_filter(inputs, layout)))
            )

        def postprocessor(output):
            if isinstance(output, ir.TensorBox):
                # prepack the weight as input to the template buffer
                # TODO(jgong5): prune the unused constants in V.graph
                # Should we implement it with constant folding in the scheduler instead?
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                assert isinstance(template_buffer, ir.CppTemplateBuffer)
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)

                W_node = new_input_nodes[1]
                assert W_node.get_name() in V.graph.constants
                W = V.graph.constants[W_node.get_name()]
                new_input_nodes[1] = W
                new_input_nodes, _ = pack_weight(
                    *normalize_shapes(*maybe_to_dense(new_input_nodes, layout))
                )
                W_packed = new_input_nodes[1]
                W_packed_constant = V.graph.add_tensor_constant(W_packed)
                template_buffer.inputs[1] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant
                )
            return output

        template = DataProcessorTemplateWrapper(
            CppPackedGemmTemplate,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
        )
        template.maybe_append_choice(choices)
        return template

    def render(  # type: ignore[override,return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        assert len(self.input_nodes) >= 2

        int8_gemm = self.input_nodes[0].get_dtype() == torch.uint8
        woq_gemm = is_woq_gemm(self.input_nodes[0], self.input_nodes[1])
        x_scale = None
        x_zp = None
        w_scale = None
        w_zp = None
        qGroupSize = None
        ScaleZP = None
        if int8_gemm:
            X, W = self.input_nodes[0], self.input_nodes[1]
            bias_idx = 2 if self.has_bias else 1
            inp = self.input_nodes[bias_idx] if self.has_bias else None
            x_scale = self.input_nodes[bias_idx + 1]
            x_zp = self.input_nodes[bias_idx + 2]
            w_scale = self.input_nodes[bias_idx + 3]
            w_zp = self.input_nodes[bias_idx + 4]
            Y = self.output_node
        elif woq_gemm:
            X, W = self.input_nodes[0], self.input_nodes[1]
            qGroupSize, ScaleZP = self.input_nodes[2], self.input_nodes[3]
            Y = self.output_node
            inp = None
        else:
            X, W = self.input_nodes[0], self.input_nodes[1]
            Y = self.output_node
            inp = self.input_nodes[2] if self.has_bias else None

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node

        template_buffer = Y
        gemm_output_buffer = template_buffer

        epilogues: List[ir.IRNode] = []
        reindexers: List[Optional[Callable[[List[Any]], List[Any]]]] = []
        epilogue_creators: List[Callable[[ir.Buffer], ir.Pointwise]] = []
        fake_buffers: List[ir.Buffer] = []
        Y_aliases: Set[str] = set()
        # TODO(jgong5): for int8 gemm, bias-add is handled outside of gemm template,
        # but we'd better move it here to align with fp.
        if inp is not None and self.beta != 0 and not int8_gemm:
            # add an epilogue for bias add
            def _bias_add_epilogue(buf):
                return create_epilogue_with_attr(
                    buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
                )

            epilogue_creators.append(_bias_add_epilogue)
        if self.epilogue_creator is not None:
            epilogue_creators.append(self.epilogue_creator)

        # NOTE [How CPP GEMM template epilogues are organized]
        #   gemm_output_buffer
        #     --> zero or more in-template epilogues (created by `epilogue_creators`) -->
        #   template_buffer
        #     --> zero or more out-of-template epilogues (`epilogue_nodes`) -->
        #   Y
        if epilogue_creators:
            gemm_output_name = "buf_GemmOut"
            gemm_output_buffer = ir.Buffer(gemm_output_name, template_buffer.layout)
            current_input_buffer = gemm_output_buffer
            for i, creator in enumerate(epilogue_creators):
                if i == len(epilogue_creators) - 1:
                    buffer_name = template_buffer.get_name()
                else:
                    buffer_name = f"buf_GemmOut_epilogue_{i}"
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        layout=template_buffer.layout,
                        data=creator(current_input_buffer),
                    )
                )
                fake_buffers.append(current_input_buffer)
                Y_aliases.add(current_input_buffer.get_name())
                reindexers.append(None)
                if i < len(epilogue_creators) - 1:
                    current_input_buffer = ir.Buffer(
                        buffer_name, template_buffer.layout
                    )

        Y_2d: Union[ir.Buffer, ir.ReinterpretView] = Y
        use_local_acc = (
            self.layout.dtype != torch.float
            or int8_gemm
            or self.padded_n != self.n
            or self.maybe_k_slicing()
        )
        if epilogue_nodes:
            epilogues.extend(epilogue_nodes)
            assert Y.get_numel() == epilogues[-1].get_numel()
            Y = cast(ir.Buffer, epilogues[-1])
            Y_aliases.add(template_buffer.get_name())
            if (
                Y.get_size() == template_buffer.get_size()
                and Y.get_stride() == template_buffer.get_stride()
            ):
                reindexers.extend([None] * len(epilogue_nodes))
                Y_2d = Y
            else:
                stride_reversed_order = list(
                    reversed(ir.get_stride_order(Y.get_stride()))
                )
                stride_reindex = ir.same_reorder(stride_reversed_order)
                ordered_size = [Y.get_size()[i] for i in stride_reversed_order]
                reshape_reindex = ir.View.dynamic_reshape_indexer(
                    ordered_size, template_buffer.get_size()
                )
                reindexer = ir.fuse_reindexing(stride_reindex, reshape_reindex)
                reindexers.extend([reindexer] * len(epilogue_nodes))  # type: ignore[list-item]
                if isinstance(Y, ir.BaseView):
                    storage = ir.StorageBox(Y.unwrap_view())
                else:
                    assert isinstance(Y, ir.Buffer)
                    storage = ir.StorageBox(Y)
                Y_2d = ir.ReinterpretView(storage, template_buffer.get_layout())

        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X.get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X.get_dtype(),
            input2_dtype=X.get_dtype() if woq_gemm else W.get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        assert micro_gemm is not None
        assert self.register_blocking == micro_gemm.register_blocking
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmAMX):
            counters["inductor"]["cpp_micro_gemm_amx_counter"] += 1

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            GemmOut=gemm_output_buffer,
            aliases={alias: Y.get_name() for alias in Y_aliases},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            epilogue_nodes=epilogues,
            reindexers=reindexers,
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            maybe_k_slicing=self.maybe_k_slicing(),
            x_scale=x_scale,
            x_zp=x_zp,
            w_scale=w_scale,
            w_zp=w_zp,
            acc_buf_dtype=torch.int32 if int8_gemm else torch.float,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            qGroupSize=qGroupSize,
            ScaleZP=ScaleZP,
            dequant_buf_dtype=X.get_dtype(),
        )
        with contextlib.ExitStack() as stack:
            for buf in fake_buffers:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)
