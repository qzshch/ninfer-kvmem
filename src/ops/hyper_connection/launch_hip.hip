#include "ninfer/ops/hyper_connection.h"
#include "core/device.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__device__ float bf(float x) {
    unsigned b=__float_as_uint(x);
    if((b&0x7fffffffu)>0x7f800000u)return __uint_as_float((b&0xffff0000u)|0x400000u);
    return __uint_as_float((b+0x7fffu+((b>>16)&1))&0xffff0000u);
}
__device__ float sigmoid(float x){return 1.f/(1.f+expf(-x));}
__global__ void coefficients_kernel(const float* projection,const float* base,const float* scale,
                                    float* output,int iterations,float eps) {
    if(threadIdx.x)return;
    const int t=blockIdx.x;const auto* p=projection+t*24;auto* o=output+t*24;
    float a[16];
    for(int i=0;i<4;++i){o[i]=sigmoid(p[i]*scale[0]+base[i])+eps;o[4+i]=2*sigmoid(p[4+i]*scale[1]+base[4+i]);}
    for(int i=0;i<4;++i){
        float m=-INFINITY;
        for(int j=0;j<4;++j){a[i*4+j]=p[8+i*4+j]*scale[2]+base[8+i*4+j];m=fmaxf(m,a[i*4+j]);}
        float sum=0;for(int j=0;j<4;++j){a[i*4+j]=expf(a[i*4+j]-m);sum+=a[i*4+j];}
        for(int j=0;j<4;++j)a[i*4+j]=a[i*4+j]/sum+eps;
    }
    for(int step=0;step<iterations;++step){
        if(step){for(int i=0;i<4;++i){float sum=eps;for(int j=0;j<4;++j)sum+=a[i*4+j];for(int j=0;j<4;++j)a[i*4+j]/=sum;}}
        for(int j=0;j<4;++j){float sum=eps;for(int i=0;i<4;++i)sum+=a[i*4+j];for(int i=0;i<4;++i)a[i*4+j]/=sum;}
    }
    for(int i=0;i<16;++i)o[8+i]=a[i];
}
__global__ void collapse_kernel(const float* residual,const float* coefficients,float* output,int hidden) {
    const int c=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(c>=hidden)return;
    const auto* r=residual+static_cast<long long>(t)*4*hidden;
    const auto* a=coefficients+t*24;
    float v=0;for(int i=0;i<4;++i)v+=a[i]*r[i*hidden+c];
    output[static_cast<long long>(t)*hidden+c]=bf(v);
}
__global__ void expand_kernel(const float* residual,const float* y,const float* coefficients,float* output,int hidden) {
    const int c=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(c>=hidden)return;
    const auto* r=residual+static_cast<long long>(t)*4*hidden;
    const auto* a=coefficients+t*24;
    // Read all streams before writing, so exact residual/output alias is valid.
    float values[4];for(int i=0;i<4;++i)values[i]=r[i*hidden+c];
    for(int j=0;j<4;++j){
        float sum=0;for(int i=0;i<4;++i)sum+=bf(a[8+i*4+j])*values[i];
        output[(static_cast<long long>(t)*4+j)*hidden+c]=bf(bf(bf(a[4+j])*y[static_cast<long long>(t)*hidden+c])+bf(sum));
    }
}
void positive(int hidden,int tokens){if(hidden<1||tokens<1||tokens>65535)throw std::invalid_argument("hyper_connection: invalid dimensions");}
}
void hyper_connection_coefficients(const float* p,const float* b,const float* s,float* o,int t,int it,float eps,cudaStream_t stream){
    positive(1,t);if(!p||!b||!s||!o||it<1||it>128||!(eps>0)||!std::isfinite(eps))throw std::invalid_argument("hyper_connection: coefficient operands");
    coefficients_kernel<<<t,1,0,stream>>>(p,b,s,o,it,eps);CUDA_CHECK(cudaGetLastError());
}
void hyper_connection_collapse(const float* r,const float* a,float* o,int h,int t,cudaStream_t stream){
    positive(h,t);if(!r||!a||!o||r==o)throw std::invalid_argument("hyper_connection: collapse operands");
    collapse_kernel<<<dim3((h+255)/256,t),256,0,stream>>>(r,a,o,h);CUDA_CHECK(cudaGetLastError());
}
void hyper_connection_expand(const float* r,const float* y,const float* a,float* o,int h,int t,cudaStream_t stream){
    positive(h,t);if(!r||!y||!a||!o||y==o)throw std::invalid_argument("hyper_connection: expand operands");
    expand_kernel<<<dim3((h+255)/256,t),256,0,stream>>>(r,y,a,o,h);CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
