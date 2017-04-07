#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)
#include <immintrin.h>
int has_intel_knl_features()
{
  const unsigned long knl_features =
      (_FEATURE_AVX512F | _FEATURE_AVX512ER |
       _FEATURE_AVX512PF | _FEATURE_AVX512CD );
  return _may_i_use_cpu_feature( knl_features );
}

int has_intel_avx2_features()
{
  const unsigned long avx2_features = _FEATURE_AVX2;
  return _may_i_use_cpu_feature( avx2_features );
}
#else /* non-Intel compiler */
#include <stdint.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
  __cpuidex(abcd, eax, ecx);
#else
  uint32_t ebx, edx;
 #if defined( __i386__ ) && defined ( __PIC__ )
  /* in case of PIC under 32-bit EBX cannot be clobbered */
  __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
 # else
  __asm__ ( "cpuid" : "+b" (ebx),
 # endif
              "+a" (eax), "+c" (ecx), "=d" (edx) );
        abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
}

int check_xcr0_zmm() {
  uint32_t xcr0;
  uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
#if defined(_MSC_VER)
  xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
#else
  __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
  return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm); /* check if xmm, zmm and zmm state are enabled in XCR0 */
}

int check_xcr0_ymm()
{
 uint32_t xcr0;
  uint32_t ymm_xmm = (1 << 2) | (1 << 1);
#if defined(_MSC_VER)
 xcr0 = (uint32_t)_xgetbv(0); /* min VS2010 SP1 compiler is required */
#else
 __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
 return ((xcr0 & ymm_xmm) == ymm_xmm); /* checking if xmm and ymm state are enabled in XCR0 */
}

int has_intel_knl_features() {
    uint32_t abcd[4];
    uint32_t osxsave_mask   =   (1 << 27); // OSX.
    uint32_t knl_bmi12_mask =   (1 << 16) | // AVX-512F
                                (1 << 26) | // AVX-512PF
                                (1 << 27) | // AVX-512ER
                                (1 << 28);  // AVX-512CD
    run_cpuid( 1, 0, abcd );
    // step 1 - must ensure OS supports extended processor state management
    if ( (abcd[2] & osxsave_mask) != osxsave_mask )
        return 0;
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & knl_bmi12_mask) != knl_bmi12_mask )
        return 0;
    // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
    if ( ! check_xcr0_zmm() )
        return 0;

    return 1;
}

int has_intel_avx2_features() {
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask )
        return 0;
    if ( ! check_xcr0_ymm() )
        return 0;
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask )
        return 0;
    return 1;
}
#endif /* non-Intel compiler */

static int can_use_intel_knl_features() {
  static int knl_features_available = -1;
  /* test is performed once */
  if (knl_features_available < 0 )
    knl_features_available = has_intel_knl_features();
  return knl_features_available;
}

static int can_use_intel_avx2_features() {
  static int avx2_features_available = -1;
  /* test is performed once */
  if (avx2_features_available < 0 )
    avx2_features_available = has_intel_avx2_features();
  return avx2_features_available;
}

int cpu_support_avx512_p()
{
    return can_use_intel_knl_features();
    // TODO the following code requires gcc 6 or above
    //__builtin_cpu_init();
    //return  true && __builtin_cpu_supports("avx512f")
    //             && __builtin_cpu_supports("avx512cd")
    //             && __builtin_cpu_supports("avx512er")
    //             && __builtin_cpu_supports("avx512pf");
    // TODO icc path not enabled
    //const unsigned long knl_features =
    //    (_FEATURE_AVX512F | _FEATURE_AVX512ER |
    //    _FEATURE_AVX512PF | _FEATURE_AVX512CD );
    //    return _may_i_use_cpu_feature( knl_features );
}

int cpu_support_avx2_p()
{
    return can_use_intel_avx2_features();
    //__builtin_cpu_init();
    //return 1 && __builtin_cpu_supports("avx2");
    // TODO icc path not enabled
    //const unsigned long avx2_features = _FEATURE_AVX2
    //    return _may_i_use_cpu_feature( avx2_features );
}


