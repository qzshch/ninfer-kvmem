// Shared private reduction for full and distributed vocabulary argmax.
__global__ void argmax_kernel(const float* x, std::int32_t* ids, int width,
                              ArgmaxPair* pairs = nullptr, int begin = 0) {
    const int t = blockIdx.x, lane = threadIdx.x;
    __shared__ float values[256];
    __shared__ int indices[256], bad[256];
    float best = -INFINITY;
    int id = 0, invalid = 0;
    for (int c = lane; c < width; c += 256) {
        float v = x[static_cast<long long>(t) * width + c];
        invalid |= !isfinite(v);
        if (v > best || (v == best && c < id)) {
            best = v;
            id   = c;
        }
    }
    values[lane]  = best;
    indices[lane] = id;
    bad[lane]     = invalid;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (lane < s) {
            bad[lane] |= bad[lane + s];
            if (values[lane + s] > values[lane] ||
                (values[lane + s] == values[lane] && indices[lane + s] < indices[lane])) {
                values[lane]  = values[lane + s];
                indices[lane] = indices[lane + s];
            }
        }
        __syncthreads();
    }
    if (!lane) {
        const int id = bad[0] ? -1 : indices[0] + begin;
        if (pairs) pairs[t] = {values[0], id};
        else ids[t] = id;
    }
    return;
}
