#include "ninfer/ops/hyper_connection.h"
#include "core/arena.h"
#include "core/device.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
double bf(double x){auto b=std::bit_cast<unsigned>(static_cast<float>(x));return std::bit_cast<float>((b+0x7fffu+((b>>16)&1))&0xffff0000u);}
double sig(double x){return 1/(1+std::exp(-x));}
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
std::array<double,24> reference(const float* p,const std::vector<float>& base,const std::vector<float>& scale,int it,double eps){
    std::array<double,24> a{};
    for(int i=0;i<4;++i){a[i]=sig(double(p[i])*scale[0]+base[i])+eps;a[4+i]=2*sig(double(p[4+i])*scale[1]+base[4+i]);}
    for(int r=0;r<4;++r){double m=-INFINITY,sum=0;for(int c=0;c<4;++c){a[8+r*4+c]=double(p[8+r*4+c])*scale[2]+base[8+r*4+c];m=std::max(m,a[8+r*4+c]);}
        for(int c=0;c<4;++c){a[8+r*4+c]=std::exp(a[8+r*4+c]-m);sum+=a[8+r*4+c];}for(int c=0;c<4;++c)a[8+r*4+c]=a[8+r*4+c]/sum+eps;}
    for(int step=0;step<it;++step){
        if(step)for(int r=0;r<4;++r){double sum=eps;for(int c=0;c<4;++c)sum+=a[8+r*4+c];for(int c=0;c<4;++c)a[8+r*4+c]/=sum;}
        for(int c=0;c<4;++c){double sum=eps;for(int r=0;r<4;++r)sum+=a[8+r*4+c];for(int r=0;r<4;++r)a[8+r*4+c]/=sum;}
    }return a;
}
double value(int i,int seed){unsigned b=static_cast<unsigned>(i+seed*13)*747796405u+2891336453u;b=((b>>((b>>28)+4))^b)*277803737u;b^=b>>22;return bf(std::bit_cast<float>((b&0x807fffffu)|((124+b%6)<<23)));}
void compare(const std::vector<float>& got,const std::vector<double>& ref,const char* label){
    double e2=0,r2=0,peak=0,err=0;
    for(std::size_t i=0;i<ref.size();++i){require(std::isfinite(got[i]),"nonfinite mHC output");auto d=std::abs(got[i]-ref[i]);e2+=d*d;r2+=ref[i]*ref[i];peak=std::max(peak,std::abs(ref[i]));err=std::max(err,d);}
    const double l2=std::sqrt(e2/std::max(r2,1e-30));
    if(l2>0.0022||err>1e-5+0.004*peak){std::fprintf(stderr,"%s l2=%.12g gross=%.12g bound=%.12g\n",label,l2,err,1e-5+.004*peak);throw std::runtime_error("mHC independent FP64 oracle criterion");}
}
void run(DeviceContext& ctx,int h,int t,int it,int mode,bool alias){
    std::vector<float> r(static_cast<std::size_t>(t)*4*h),y(static_cast<std::size_t>(t)*h),p(t*24),base(24),scale{.71f,-1.13f,.93f};
    for(std::size_t i=0;i<r.size();++i)r[i]=value(i,1);for(std::size_t i=0;i<y.size();++i)y[i]=value(i,2);
    for(std::size_t i=0;i<p.size();++i)p[i]=mode==2?0:static_cast<float>(value(i,3)*(mode?90:1));
    for(int i=0;i<24;++i)base[i]=mode==2?0:static_cast<float>(value(i,4));
    if(mode==2){std::fill(r.begin(),r.end(),0);std::fill(y.begin(),y.end(),0);}
    DeviceBuffer dr(r.size()*4),dy(y.size()*4),dp(p.size()*4),db(96),ds(12),da(p.size()*4),collapsed(y.size()*4+32),expanded(r.size()*4+32);
    dr.copy_from_host(r.data(),dr.bytes);dy.copy_from_host(y.data(),dy.bytes);dp.copy_from_host(p.data(),dp.bytes);db.copy_from_host(base.data(),db.bytes);ds.copy_from_host(scale.data(),ds.bytes);collapsed.fill(0xa5);expanded.fill(0xa5);
    auto* co=reinterpret_cast<float*>(static_cast<std::byte*>(collapsed.p)+16);
    auto* ex=alias?static_cast<float*>(dr.p):reinterpret_cast<float*>(static_cast<std::byte*>(expanded.p)+16);
    ops::hyper_connection_coefficients(static_cast<float*>(dp.p),static_cast<float*>(db.p),static_cast<float*>(ds.p),static_cast<float*>(da.p),t,it,1e-6f,ctx.stream);
    ops::hyper_connection_collapse(static_cast<float*>(dr.p),static_cast<float*>(da.p),co,h,t,ctx.stream);
    ops::hyper_connection_expand(static_cast<float*>(dr.p),static_cast<float*>(dy.p),static_cast<float*>(da.p),ex,h,t,ctx.stream);ctx.synchronize();
    std::vector<float> a(p.size()),oc(y.size()),oe(r.size());da.copy_to_host(a.data(),da.bytes);CUDA_CHECK(cudaMemcpy(oc.data(),co,oc.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(oe.data(),ex,oe.size()*4,cudaMemcpyDeviceToHost));
    std::vector<double> rc(y.size()),re(r.size());
    for(int token=0;token<t;++token){auto ref=reference(p.data()+token*24,base,scale,it,1e-6f);
        for(int i=0;i<24;++i)require(std::abs(a[token*24+i]-ref[i])<=1e-5*(1+std::abs(ref[i])),"mHC coefficients FP64 bound");
        for(int c=0;c<h;++c){double sum=0;for(int i=0;i<4;++i)sum+=ref[i]*r[(token*4+i)*h+c];rc[token*h+c]=sum;
            for(int j=0;j<4;++j){double mix=0;for(int i=0;i<4;++i)mix+=bf(ref[8+i*4+j])*r[(token*4+i)*h+c];re[(token*4+j)*h+c]=bf(bf(ref[4+j])*y[token*h+c])+bf(mix);}}
    }
    compare(oc,rc,"collapse");compare(oe,re,"expand");
    for(auto* buffer:{&collapsed,&expanded}){std::vector<std::byte> bytes(buffer->bytes);buffer->copy_to_host(bytes.data(),bytes.size());for(int i=0;i<16;++i)require(bytes[i]==std::byte{0xa5}&&bytes[bytes.size()-1-i]==std::byte{0xa5},"mHC guard changed");}
    if(!alias){std::vector<float> original(r.size());dr.copy_to_host(original.data(),dr.bytes);require(original==r,"mHC disjoint source changed");}
}
}
int main(){try{DeviceContext ctx;int count=0;for(int h:{1,7,257,4096})for(int t:{1,8})for(int mode:{0,1,2})for(bool alias:{false,true}){run(ctx,h,t,20,mode,alias);++count;}run(ctx,4096,8,1,0,false);++count;std::printf("PASS: %d mHC coefficient/collapse/expand fixtures; FP64 oracle, BF16 semantic boundaries, alias and guards\n",count);return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
