
// -*- c++ -*-

#include "FaissHookFvec.h"

#include <iostream>
#include <mutex>

#include "distances_simd.h"
// #include "distances_simd_avx.h"
// #include "distances_simd_avx512.h"
// #include "distances_simd_sse.h"
#ifdef __linux__
#include "instruction_set.h"
#endif

namespace faiss {

bool faiss_use_avx512 = false;
bool faiss_use_avx2 = false;
bool faiss_use_sse4_2 = false;

/* set default to AVX */
fvec_func_ptr fvec_inner_product = fvec_inner_product_ref;
fvec_func_ptr fvec_L2sqr = fvec_L2sqr_ref;
fvec_func_ptr fvec_L1 = fvec_L1_ref;
fvec_func_ptr fvec_Linf = fvec_Linf_ref;
fvec_norm_L2sqr_func_ptr fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
fvec_L2sqr_ny_func_ptr fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
fvec_inner_products_ny_func_ptr fvec_inner_products_ny = fvec_inner_products_ny_ref;
fvec_madd_func_ptr fvec_madd = fvec_madd_ref;
fvec_madd_and_argmin_func_ptr fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

void
hook_fvec(std::string& simd_type) {
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);

#ifdef __linux__

    /* for IVFFLAT */
    fvec_inner_product = fvec_inner_product_ref;
    fvec_L2sqr = fvec_L2sqr_ref;
    fvec_L1 = fvec_L1_ref;
    fvec_Linf = fvec_Linf_ref;

    fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
    fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
    fvec_inner_products_ny = fvec_inner_products_ny_ref;
    fvec_madd = fvec_madd_ref;
    fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

    simd_type = "REF";
#else
    simd_type = "REF";
#endif
}
#ifdef __linux__
bool
cpu_support_avx512() {
    return false;
}

bool
cpu_support_avx2() {
    return false;
}

bool
cpu_support_sse4_2() {
    return false;
}
#endif
}  // namespace faiss
