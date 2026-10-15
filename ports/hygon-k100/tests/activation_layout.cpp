#include "models/glm5_next/math.h"
#include "ninfer/ops/resident_transform.h"
#include "core/device.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>
#include <stdexcept>

using namespace ninfer;
unsigned short encode(float v) {
    const auto b = std::bit_cast<unsigned>(v);
    return static_cast<unsigned short>((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
float decode(unsigned short v) { return std::bit_cast<float>(static_cast<unsigned>(v) << 16); }
int main() try {
    DeviceContext device;
    for (int width : {512,1536,4096}) for (int rows : {1,3}) {
        std::vector<float> x(width * rows), gain(width);
        std::vector<unsigned short> words(x.size());
        for (int i = 0; i < width * rows; ++i) {
            words[i] = encode(std::ldexp(static_cast<float>(i % 9 - 4), -18)); x[i] = decode(words[i]);
        }
        for (int c = 0; c < width; ++c) gain[c] = .5f + (c % 7) * .125f;
        DeviceBuffer input(x.size()*4), compact(words.size()*2), g(gain.size()*4),
                     control(x.size()*4), output(words.size()*2+32);
        input.copy_from_host(x.data(),input.bytes); compact.copy_from_host(words.data(),compact.bytes);
        g.copy_from_host(gain.data(),g.bytes); output.fill(0xa5);
        Tensor weight(g.p,DType::FP32,{width}), bf(compact.p,DType::BF16,{width,rows});
        Tensor y(static_cast<std::byte*>(output.p)+16,DType::BF16,{width,rows});
        ops::resident_norm(static_cast<float*>(input.p),&weight,nullptr,nullptr,static_cast<float*>(control.p),
                           width,rows,models::glm5_next::rms_norm_epsilon,false,true,device.stream);
        ops::resident_norm(bf,&weight,nullptr,nullptr,y,models::glm5_next::rms_norm_epsilon,false,device.stream);
        device.synchronize();
        std::vector<unsigned short> actual(words.size()+16); output.copy_to_host(actual.data(),output.bytes);
        std::vector<float> reference(x.size()); control.copy_to_host(reference.data(),control.bytes);
        for (int k = 0; k < 8; ++k)
            if (actual[k]!=0xa5a5 || actual[actual.size()-1-k]!=0xa5a5) throw std::runtime_error("norm guards");
        for (int row = 0; row < rows; ++row) {
            double sum = 0; for (int c = 0; c < width; ++c) sum += double(x[row*width+c])*x[row*width+c];
            const double inv = 1/std::sqrt(sum/width + 1e-5);
            for (int c = 0; c < width; ++c) {
                const int i = row*width+c;
                const double expected = double(x[i])*inv*gain[c];
                const double floor = std::abs(decode(encode(static_cast<float>(expected)))-expected);
                const float value = decode(actual[i+8]);
                if (std::bit_cast<unsigned>(value)!=std::bit_cast<unsigned>(reference[i]) || !std::isfinite(value) || std::abs(value-expected)>floor+4e-6*std::abs(expected)+1e-30)
                    throw std::runtime_error("model RMS epsilon or physical activation boundary changed");
            }
        }
    }
    std::puts("PASS: model RMSNorm FP64 oracle and exact FP32/BF16 storage boundary, including small MLA inputs");
} catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
