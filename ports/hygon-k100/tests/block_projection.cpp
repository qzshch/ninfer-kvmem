#include "ninfer/ops/linear.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "artifact/formats.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
void require(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
template<class T>std::vector<T> read(const std::filesystem::path& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"missing CPU codec fixture");
    auto bytes=static_cast<std::size_t>(f.tellg());require(bytes%sizeof(T)==0,"unaligned fixture bytes");
    std::vector<T> out(bytes/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),bytes);
    require(bool(f),"truncated fixture read");return out;
}
std::uint16_t bf16(float v){auto b=std::bit_cast<unsigned>(v);return (b+0x7fffu+((b>>16)&1))>>16;}
double unbf16(std::uint16_t b){return std::bit_cast<float>(static_cast<unsigned>(b)<<16);}

double run(DeviceContext& ctx,const std::filesystem::path& root,const nlohmann::json& item,int tokens,bool bf) {
    const auto raw=read<std::byte>(root/item.at("encoded").get<std::string>());
    const auto oracle=read<double>(root/item.at("oracle").get<std::string>());
    const int n=item.at("n"),k=item.at("k"),rows=item.at("parent_rows"),begin=item.at("row_begin");
    const auto type=artifact::parse_format(item.at("format").get<std::string>());
    const std::array<std::uint64_t,2> shape{static_cast<unsigned>(rows),static_cast<unsigned>(k)};
    const auto geometry=weight_geometry(type,QuantLayout::BlockInterleaved,shape);
    require(geometry.bytes==raw.size()&&oracle.size()==static_cast<std::size_t>(rows)*k,"codec/geometry mismatch");
    DeviceBuffer weights(raw.size());weights.copy_from_host(raw.data(),raw.size());
    WeightParent parent{geometry,static_cast<const std::byte*>(weights.p)};
    auto w=native_weight(WeightView{{static_cast<unsigned>(n),static_cast<unsigned>(k)},
        {{&parent,static_cast<std::uint64_t>(begin)*k,static_cast<std::uint64_t>(begin+n)*k}}});
    std::vector<float> x(static_cast<std::size_t>(k)*tokens);std::vector<std::uint16_t> xb(x.size());
    for(std::size_t i=0;i<x.size();++i){
        unsigned bits=static_cast<unsigned>(i)*747796405u+2891336453u;
        bits=((bits>>((bits>>28)+4))^bits)*277803737u;bits^=bits>>22;
        x[i]=std::bit_cast<float>((bits&0x807fffffu)|((124+(bits%6))<<23));
        xb[i]=bf16(x[i]);if(bf)x[i]=static_cast<float>(unbf16(xb[i]));
    }
    const auto word=bf?2:4;DeviceBuffer input(x.size()*word),output(static_cast<std::size_t>(n)*tokens*word+32);
    input.copy_from_host(bf?static_cast<const void*>(xb.data()):x.data(),input.bytes);output.fill(0xa5);
    const auto dtype=bf?DType::BF16:DType::FP32;
    Tensor a(input.p,dtype,{k,tokens}),y(static_cast<std::byte*>(output.p)+16,dtype,{n,tokens});
    if(bf)ops::linear(a,w,y,ctx.stream);else ops::linear_fp32(a,w,y,ctx.stream);
    ctx.synchronize();auto observed=std::vector<std::byte>(output.bytes);output.copy_to_host(observed.data(),observed.size());
    for(int i=0;i<16;++i)require(observed[i]==std::byte{0xa5}&&observed[observed.size()-1-i]==std::byte{0xa5},"block projection output guard");
    std::vector<std::byte> after(raw.size());weights.copy_to_host(after.data(),after.size());require(after==raw,"encoded parent changed");
    double worst=0;
    for(int t=0;t<tokens;++t)for(int row=0;row<n;++row){
        double expected=0,absolute=0;
        for(int c=0;c<k;++c){auto p=oracle[static_cast<std::size_t>(begin+row)*k+c]*x[static_cast<std::size_t>(t)*k+c];expected+=p;absolute+=std::abs(p);}
        double value,floor=0;const auto offset=16+(static_cast<std::size_t>(t)*n+row)*word;
        if(bf){std::uint16_t v;std::memcpy(&v,observed.data()+offset,2);value=unbf16(v);
            const auto approx=static_cast<int>(bf16(static_cast<float>(expected)));floor=std::abs(unbf16(approx)-expected);
            for(int d:{-1,1})if(approx+d>=0&&approx+d<=65535)floor=std::min(floor,std::abs(unbf16(approx+d)-expected));
        }else{float v;std::memcpy(&v,observed.data()+offset,4);value=v;}
        const auto ratio=std::max(0.,std::abs(value-expected)-floor)/(4e-6*absolute+1e-30);worst=std::max(worst,ratio);
        if(!std::isfinite(value)||ratio>1){std::fprintf(stderr,"format=%s row=%d token=%d expected=%g value=%g ratio=%g\n",item.at("format").get<std::string>().c_str(),row,t,expected,value,ratio);throw std::runtime_error("block projection FP64 full-dot criterion");}
    }
    return worst;
}
}
int main(int argc,char** argv)try {
    require(argc==2,"fixture directory required");std::filesystem::path root(argv[1]);
    std::ifstream f(root/"manifest.json");nlohmann::json cases;f>>cases;DeviceContext ctx;unsigned count=0;double worst=0;
    for(const auto& item:cases)for(int tokens:{1,8})for(bool bf:{false,true}) {
        worst=std::max(worst,run(ctx,root,item,tokens,bf));++count;
    }
    std::printf("PASS: %u block projection cases; five source formats; independent scalar CPU block decode and full FP64 dot; max_arithmetic_budget_fraction=%.9g\n",count,worst);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
