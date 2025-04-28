#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

// #include <ATen/cpu/vec/intrinsics.h>
// #include <ATen/cpu/vec/vec_base.h>
// #if (defined(CPU_CAPABILITY_AVX512))
// #define SLEEF_STATIC_LIBS
// #include <sleef.h>
// #endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
template <typename T>
class Vectorizedf8 {
private:
  __m512i values;
public:
  using value_type = uint8_t;
  using size_type = int;
  static constexpr size_type size() {
    return 64;
  }
  Vectorizedf8() {}
  Vectorizedf8(__m512i v) : values(v) {}
  Vectorizedf8(T val) {
    value_type uw = val.x;
    // values = _mm512_set1_epi16(uw);
  }
  operator __m512i() const {
    return values;
  }
  T& operator[](int idx) = delete;
  const T& operator[](int idx) const  = delete;
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    std::cout<<"---- hit fp8 load ----"<<std::endl;
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else if (count == 16) {
      // Fast path if only load element number of 16
      __m128i input_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
      return _mm512_castsi128_si512(input_128);
    } else {
      __mmask64 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi8(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    std::cout<<"---- hit fp8 store ----"<<std::endl;
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      if (count == 16) {
        // Fast path if only store element number of 16
        _mm_storeu_si128(
          reinterpret_cast<__m128i*>(ptr),
          _mm512_castsi512_si128(values));
      } else {
        __mmask64 mask = (1ULL << count) - 1;
        _mm512_mask_storeu_epi8(ptr, mask, values);
      }
    }
  }

  Vectorized<T> abs() const {
    __m512i v;
    return v;
  }


private:

};


template <>
class Vectorized<Float8_e4m3fn>: public Vectorizedf8<Float8_e4m3fn> {
public:
  using Vectorizedf8::Vectorizedf8;

  using value_type = Float8_e4m3fn;

  Vectorized<Float8_e4m3fn> frac() const;
  Vectorized<Float8_e4m3fn> eq(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> ne(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> gt(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> ge(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> lt(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> le(const Vectorized<Float8_e4m3fn>& other) const;
};

Vectorized<Float8_e4m3fn> inline operator+(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator-(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator*(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator/(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator&(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator|(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}
Vectorized<Float8_e4m3fn> inline operator^(const Vectorized<Float8_e4m3fn>& a, const Vectorized<Float8_e4m3fn>& b) {
  return a;
}

#endif

}}
