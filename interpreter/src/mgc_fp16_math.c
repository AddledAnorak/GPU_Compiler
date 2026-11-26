#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "mgc_interpreter.h"

// =============================================================================
// IEEE-754 Half-Precision (FP16) Software Conversion
// =============================================================================

uint16_t mgc_float_to_fp16(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t  exp  = ((u.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u.u & 0x7FFFFF;

    if (exp <= 0) {
        // Subnormal or underflow
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        // Infinity or NaN
        if (mant != 0) return (uint16_t)(sign | 0x7E00); // qNaN
        return (uint16_t)(sign | 0x7C00); // Inf
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

float mgc_fp16_to_float(uint16_t h) {
    union { float f; uint32_t u; } u;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            u.u = sign;
            return u.f;
        }
        // Subnormal
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
        u.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        return u.f;
    } else if (exp == 31) {
        // Inf or NaN
        u.u = sign | 0x7F800000 | (mant << 13);
        return u.f;
    }

    u.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return u.f;
}

// =============================================================================
// FP16 Arithmetic Primitives
// =============================================================================

uint16_t mgc_fp16_add(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    return mgc_float_to_fp16(fa + fb);
}

uint16_t mgc_fp16_sub(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    return mgc_float_to_fp16(fa - fb);
}

uint16_t mgc_fp16_mul(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    return mgc_float_to_fp16(fa * fb);
}

uint16_t mgc_fp16_div(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    if (fb == 0.0f) {
        uint16_t sign = (a ^ b) & 0x8000;
        return (uint16_t)(sign | 0x7C00); // Inf
    }
    return mgc_float_to_fp16(fa / fb);
}

uint16_t mgc_fp16_min(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    return (fa < fb) ? a : b;
}

uint16_t mgc_fp16_max(uint16_t a, uint16_t b) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    return (fa > fb) ? a : b;
}

uint16_t mgc_fp16_sqrt(uint16_t a) {
    float fa = mgc_fp16_to_float(a);
    if (fa < 0.0f) {
        return 0x7E00; // Quiet NaN for negative sqrt
    }
    return mgc_float_to_fp16(sqrtf(fa));
}

uint16_t mgc_fp16_rsqrt(uint16_t a) {
    float fa = mgc_fp16_to_float(a);
    if (fa <= 0.0f) {
        return 0x7E00; // Quiet NaN for non-positive rsqrt
    }
    return mgc_float_to_fp16(1.0f / sqrtf(fa));
}

uint16_t mgc_fp16_round(uint16_t a, uint8_t sel) {
    float f = mgc_fp16_to_float(a);
    float res;
    switch (sel & 0x03) {
        case 0: res = truncf(f); break; // trunc
        case 1: res = roundf(f); break; // round
        case 2: res = ceilf(f);  break; // ceil
        case 3: res = floorf(f); break; // floor
        default: res = f; break;
    }
    return mgc_float_to_fp16(res);
}

bool mgc_fp16_compare(uint16_t a, uint16_t b, uint8_t op) {
    float fa = mgc_fp16_to_float(a);
    float fb = mgc_fp16_to_float(b);
    bool a_nan = isnan(fa);
    bool b_nan = isnan(fb);

    switch (op & 0x07) {
        case 0: return (!a_nan && !b_nan) && (fa < fb);
        case 1: return (!a_nan && !b_nan) && (fa <= fb);
        case 2: return (!a_nan && !b_nan) && (fa == fb);
        case 3: return (!a_nan && !b_nan) && (fa >= fb);
        case 4: return (!a_nan && !b_nan) && (fa > fb);
        case 5: return (!a_nan && !b_nan) && (fa != fb);
        case 6: return (!a_nan && !b_nan); // ordered
        case 7: return (a_nan || b_nan);   // unordered
        default: return false;
    }
}
