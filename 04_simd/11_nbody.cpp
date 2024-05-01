#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>


// Horizontal reduction of m256 register by Marat Dukhan
// see: https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

// Util. for printing SIMD vectors
void dump(__m256 vec, const char *msg) {
    float output[8];
    _mm256_storeu_ps(output, vec);
    printf("%s\n", msg);
    for (int i = 0; i < 8; i++) {
        printf("%d: %f\n", i, output[i]);
    }
}


int main() {
  //srand48(10); // for testing
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  // Note: the slides for the assigment said that 
  // I should use the larger _m512 registers but since we are only dealing 
  // with 8 single precision floats I'll stick to the smaller __m256, 
  // hope that's okay!
  const auto x_vec  = _mm256_load_ps(x);
  const auto y_vec  = _mm256_load_ps(y);
  const auto m_vec  = _mm256_load_ps(m);

  for(int i=0; i<N; i++) {
    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }

    // Load position of current body i in SIMD register
    const auto current_x = _mm256_set1_ps(x[i]);
    const auto current_y = _mm256_set1_ps(y[i]);

    // positional difference in x and y direction 
    // between body i and all other bodies
    const auto diff_x_vec = _mm256_sub_ps(current_x, x_vec);
    const auto diff_y_vec = _mm256_sub_ps(current_y, y_vec);

    // inverse radius
    const auto sum = _mm256_add_ps(
      _mm256_mul_ps(diff_x_vec, diff_x_vec),
      _mm256_mul_ps(diff_y_vec, diff_y_vec)
    );
    const auto inv_radius_vec = _mm256_rsqrt14_ps(sum);
    // mass * (1/radius) * (1/radius) * (1/radius) 
    const auto product_vec = _mm256_mul_ps(
      m_vec,
      _mm256_mul_ps(inv_radius_vec, _mm256_mul_ps(inv_radius_vec, inv_radius_vec)
    ));

    // We want to ignore i = j
    __mmask8 mask = 0;
    mask |= 1 << i;
    const auto masked_product_vec = _mm256_mask_blend_ps(mask, product_vec, _mm256_set1_ps(0));
    // f_x * mass * (1/radius) * (1/radius) * (1/radius) 
    const auto current_fx_vec = _mm256_mul_ps(diff_x_vec, masked_product_vec);
    // f_y * mass * (1/radius) * (1/radius) * (1/radius)
    const auto current_fy_vec = _mm256_mul_ps(diff_y_vec, masked_product_vec);

    // finally, save the forces for body i 
    // Side note, I guess one of the main downsides with using the 256 bit registers 
    // over the 512 ones is that Intel does not offer an equivalent to _mm512_reduce_add_ps
    fx[i] = sum8(current_fx_vec);
    fy[i] = sum8(current_fy_vec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
