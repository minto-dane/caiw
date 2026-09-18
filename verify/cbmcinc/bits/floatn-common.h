/* CBMC shadow: glibc 2.41 exposes _FloatN decls CBMC 6.6 cannot parse.
 * All __HAVE_FLOAT* forced 0 → strtof32 & friends are #if'd out.
 * caiw uses none of these interfaces. */
#define __HAVE_FLOAT16 0
#define __HAVE_FLOAT32 0
#define __HAVE_FLOAT64 0
#define __HAVE_FLOAT32X 0
#define __HAVE_FLOAT128X 0
#define __HAVE_FLOAT128 0
#define __HAVE_FLOAT64X 0
#define __HAVE_DISTINCT_FLOAT16 0
#define __HAVE_DISTINCT_FLOAT32 0
#define __HAVE_DISTINCT_FLOAT64 0
#define __HAVE_DISTINCT_FLOAT32X 0
#define __HAVE_DISTINCT_FLOAT64X 0
#define __HAVE_DISTINCT_FLOAT128X 0
#define __HAVE_DISTINCT_FLOAT128 0
#define __HAVE_FLOATN_NOT_TYPEDEF 0
#define __f32(x) 0
#define __f64(x) 0
#define __f32x(x) 0
#define __f64x(x) 0
#define __f128(x) 0
#define __f128x(x) 0
#define __CFLOAT32 float
#define __CFLOAT64 double
#define __CFLOAT32X double
#define __CFLOAT64X double
#define __CFLOAT128 long double
#define __CFLOAT128X long double
