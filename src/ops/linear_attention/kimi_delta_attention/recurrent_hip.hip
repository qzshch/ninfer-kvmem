// Native DTK implementation of the public NInfer KDA recurrence. Apache-2.0.
// Eight value rows per group; reductions use shared memory and no warp ABI.
#include "ops/linear_attention/kimi_delta_attention/launch.h"

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
__device__ float from_bf16(std::uint16_t word) {
    return __uint_as_float(static_cast<unsigned>(word) << 16);
}
__device__ std::uint16_t to_bf16(float value) {
    auto word=__float_as_uint(value);
    if ((word&0x7fffffffu)>0x7f800000u) return static_cast<std::uint16_t>((word>>16)|0x40);
    return static_cast<std::uint16_t>((word+0x7fffu+((word>>16)&1))>>16);
}
__device__ float sigmoid(float value) { return 1.f/(1.f+expf(-value)); }

template<bool Pooled>
__global__ void recurrence(Arguments a,const std::int32_t* slots) {
    constexpr int D=128,V=8;
    const int h=blockIdx.x,vbase=blockIdx.y*V,batch=blockIdx.z,c=threadIdx.x;
    const int qh=h/(a.value_heads/a.qk_heads);
    const std::int64_t state_slot=Pooled?slots[batch]:0;
    const std::int64_t state_base=(state_slot*a.value_heads+h)*D*D+vbase*D+c;
    __shared__ float q[D],k[D],reduce[V][D];
    __shared__ float beta,inv_q,inv_k;
    float state[V];
    for(int v=0;v<V;++v) state[v]=a.state_in[state_base+v*D];
    const float rate=expf(a.a_log[h]);
    for(int t=0;t<a.tokens;++t) {
        const std::int64_t qbase=((static_cast<std::int64_t>(batch)*a.tokens+t)*a.qk_heads+qh)*D;
        const std::int64_t vstart=((static_cast<std::int64_t>(batch)*a.tokens+t)*a.value_heads+h)*D;
        const float qi=from_bf16(a.q[qbase+c]),ki=from_bf16(a.k[qbase+c]);
        reduce[0][c]=qi*qi;reduce[1][c]=ki*ki;
        __syncthreads();
        for(int distance=D/2;distance;distance/=2) {
            if(c<distance){reduce[0][c]+=reduce[0][c+distance];reduce[1][c]+=reduce[1][c+distance];}
            __syncthreads();
        }
        if(c==0) {
            inv_q=rsqrtf(reduce[0][0]+kQkL2NormEps);
            inv_k=rsqrtf(reduce[1][0]+kQkL2NormEps);
            beta=sigmoid(from_bf16(a.beta[(static_cast<std::int64_t>(batch)*a.tokens+t)*a.value_heads+h]));
        }
        __syncthreads();
        q[c]=qi*inv_q;k[c]=ki*inv_k;
        const float gate=from_bf16(a.g[vstart+c])+a.dt_bias[h*D+c];
        const float alpha=expf(a.lower_bound*sigmoid(rate*gate));
        for(int v=0;v<V;++v) {state[v]*=alpha;reduce[v][c]=state[v]*k[c];}
        __syncthreads();
        for(int distance=D/2;distance;distance/=2) {
            if(c<distance)for(int v=0;v<V;++v)reduce[v][c]+=reduce[v][c+distance];
            __syncthreads();
        }
        // All threads must consume prediction scalars before any overwrite.
        float delta[V];
        for(int v=0;v<V;++v)delta[v]=beta*(from_bf16(a.v[vstart+vbase+v])-reduce[v][0]);
        __syncthreads();
        for(int v=0;v<V;++v) {state[v]+=k[c]*delta[v];reduce[v][c]=state[v]*q[c];}
        __syncthreads();
        for(int distance=D/2;distance;distance/=2) {
            if(c<distance)for(int v=0;v<V;++v)reduce[v][c]+=reduce[v][c+distance];
            __syncthreads();
        }
        if(c<V)a.out[vstart+vbase+c]=to_bf16(reduce[c][0]*a.scale);
        __syncthreads();
    }
    for(int v=0;v<V;++v)a.state_out[state_base+v*D]=state[v];
    return;
}
} // namespace

void launch_recurrent(const Arguments& args,cudaStream_t stream) {
    recurrence<false><<<dim3(args.value_heads,16,1),128,0,stream>>>(args,nullptr);
    CUDA_CHECK(cudaGetLastError());
    return;
}
void launch_batch_update(const Arguments& args,const std::int32_t* slots,int batch,cudaStream_t stream) {
    recurrence<true><<<dim3(args.value_heads,16,batch),128,0,stream>>>(args,slots);
    CUDA_CHECK(cudaGetLastError());
    return;
}
} // namespace ninfer::ops::detail::kimi_delta_attention
