// Q4 weights stay packed in global memory. Activations and KV use BF16 bits;
// reductions and DeltaNet states use FP32. No FP8/FP4 hardware is required.
typedef unsigned short B;
__device__ float f(B x) { return __uint_as_float((unsigned int)x << 16); }
__device__ B b(float x) {
    unsigned int u = __float_as_uint(x);
    return (B)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ float sig(float x) { return 1.f / (1.f + expf(-x)); }
__device__ float silu(float x) { return x * sig(x); }
__device__ float warp_sum(float x) {
    for (int d=16; d; d/=2) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}
__device__ float qweight(const unsigned char* w, const float* s, int r, int c, int k, int g) {
    int code = (w[r*((k+1)/2)+c/2] >> ((c&1)*4)) & 15;
    int scale = 2*(r*((k+g-1)/g)+c/g);
    return f(b(code*s[scale]+s[scale+1]));
}
extern "C" __global__ void qlinear(const B* x, const unsigned char* w, const float* s, B* y, int m, int k, int g) {
    int r=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32, t=blockIdx.y;
    float sum=0.f;
    if(r<m) for(int c=lane;c<k;c+=32) sum += f(x[t*k+c])*qweight(w,s,r,c,k,g);
    sum=warp_sum(sum);
    if(r<m && lane==0) y[t*m+r]=b(sum);
}
extern "C" __global__ void dense_linear(const B* x, const float* w, B* y, int m, int k) {
    int r=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x%32, t=blockIdx.y;
    float sum=0.f;
    if(r<m) for(int c=lane;c<k;c+=32) sum += f(x[t*k+c])*w[r*k+c];
    sum=warp_sum(sum);
    if(r<m && lane==0) y[t*m+r]=b(sum);
}
extern "C" __global__ void embedding(const unsigned int* ids, const unsigned char* w, const float* s, B* y, int k, int g) {
    int c=blockIdx.x*blockDim.x+threadIdx.x, t=blockIdx.y;
    if(c<k) y[t*k+c]=b(qweight(w,s,ids[t],c,k,g));
}
extern "C" __global__ void dense_embedding(const unsigned int* ids, const float* w, B* y, int k) {
    int c=blockIdx.x*blockDim.x+threadIdx.x, t=blockIdx.y;
    if(c<k) y[t*k+c]=b(w[ids[t]*k+c]);
}
extern "C" __global__ void rms(const B* x, const float* w, B* y, int width, float eps, int centered) {
    __shared__ float sums[256];
    int row=blockIdx.x, tid=threadIdx.x;
    float sum=0;
    for(int c=tid;c<width;c+=256) { float a=f(x[row*width+c]); sum+=a*a; }
    sums[tid]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) sums[tid]+=sums[tid+d]; __syncthreads(); }
    float inv=rsqrtf(sums[0]/width+eps);
    for(int c=tid;c<width;c+=256) {
        float norm=f(x[row*width+c])*inv;
        if(!centered) norm=f(b(norm));
        y[row*width+c]=b(norm*(w[c]+(centered?1.f:0.f)));
    }
}
extern "C" __global__ void add(B* x, const B* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=b(f(x[i])+f(y[i]));
}
extern "C" __global__ void swiglu(const B* gate, const B* up, B* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=b(f(b(silu(f(gate[i]))))*f(up[i]));
}
extern "C" __global__ void split_gate(const B* input, B* q, B* gate, int heads, int dim, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*heads*dim) {
        int h=i/dim,c=i%dim;
        q[i]=input[h*dim*2+c]; gate[i]=input[h*dim*2+dim+c];
    }
}
extern "C" __global__ void rope(B* x, int heads, int dim, int rotary, int start, float theta, int n) {
    int pair=blockIdx.x*blockDim.x+threadIdx.x;
    if(pair<n*heads*(rotary/2)) {
        int j=pair%(rotary/2), h=pair/(rotary/2), t=h/heads;
        float angle=(start+t)*powf(theta,-2.f*j/rotary);
        float co=f(b(cosf(angle))), si=f(b(sinf(angle)));
        int p=h*dim+j;
        float a=f(x[p]), z=f(x[p+rotary/2]);
        x[p]=b(f(b(a*co))-f(b(z*si)));
        x[p+rotary/2]=b(f(b(z*co))+f(b(a*si)));
    }
}
extern "C" __global__ void cache_write(const B* k, const B* v, B* kc, B* vc, int width, int start, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*width) { kc[start*width+i]=k[i]; vc[start*width+i]=v[i]; }
}
extern "C" __global__ void attention(const B* q, const B* kc, const B* vc, const B* gate, B* out,
    int heads, int kvheads, int dim, int start, int gated) {
    extern __shared__ float sm[];
    float* scores=sm+256;
    int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x,len=start+t+1,kh=h/(heads/kvheads);
    float local=-1.e30f;
    for(int p=tid;p<len;p+=256) {
        float sum=0;
        for(int d=0;d<dim;d++) sum+=f(q[(t*heads+h)*dim+d])*f(kc[(p*kvheads+kh)*dim+d]);
        scores[p]=f(b(f(b(sum))*rsqrtf((float)dim))); local=fmaxf(local,scores[p]);
    }
    sm[tid]=local; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) sm[tid]=fmaxf(sm[tid],sm[tid+d]); __syncthreads(); }
    float mx=sm[0]; __syncthreads();
    float sum=0;
    for(int p=tid;p<len;p+=256) { scores[p]=expf(scores[p]-mx); sum+=scores[p]; }
    sm[tid]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) sm[tid]+=sm[tid+d]; __syncthreads(); }
    for(int d=tid;d<dim;d+=256) {
        float z=0;
        for(int p=0;p<len;p++) z+=f(b(scores[p]/sm[0]))*f(vc[(p*kvheads+kh)*dim+d]);
        int i=(t*heads+h)*dim+d;
        float value=f(b(z));
        if(gated) value*=f(b(sig(f(gate[i]))));
        out[i]=b(value);
    }
}
extern "C" __global__ void conv(const B* x, const float* w, B* history, B* y, int channels, int kernel, int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<channels) for(int t=0;t<n;t++) {
        for(int j=0;j<kernel-1;j++) history[c*kernel+j]=history[c*kernel+j+1];
        history[c*kernel+kernel-1]=x[t*channels+c];
        float sum=0;
        for(int j=0;j<kernel;j++) sum+=f(history[c*kernel+j])*w[c*kernel+j];
        y[t*channels+c]=b(silu(sum));
    }
}
extern "C" __global__ void delta(const B* qkv, const B* a, const B* beta, const float* alog, const float* dt,
    float* state, B* out, int kh, int vh, int kd, int vd, int n) {
    __shared__ float qq[256],kk[256],ss[256];
    int h=blockIdx.x,tid=threadIdx.x,head=h/(vh/kh),channels=kh*kd*2+vh*vd;
    for(int t=0;t<n;t++) {
        float q=tid<kd?f(qkv[t*channels+head*kd+tid]):0;
        float k=tid<kd?f(qkv[t*channels+kh*kd+head*kd+tid]):0;
        ss[tid]=q*q; __syncthreads();
        for(int d=128;d;d/=2) { if(tid<d) ss[tid]+=ss[tid+d]; __syncthreads(); }
        qq[tid]=q*rsqrtf(ss[0]+1.e-6f)*rsqrtf((float)kd); __syncthreads();
        ss[tid]=k*k; __syncthreads();
        for(int d=128;d;d/=2) { if(tid<d) ss[tid]+=ss[tid+d]; __syncthreads(); }
        kk[tid]=k*rsqrtf(ss[0]+1.e-6f); __syncthreads();
        float step=f(a[t*vh+h])+dt[h];
        float soft=step>20.f?step:log1pf(expf(step));
        float decay=expf(-expf(alog[h])*soft), bt=f(b(sig(f(beta[t*vh+h]))));
        for(int v=tid;v<vd;v+=256) {
            float mem=0;
            for(int j=0;j<kd;j++) mem+=state[(h*kd+j)*vd+v]*decay*kk[j];
            float dv=(f(qkv[t*channels+2*kh*kd+h*vd+v])-mem)*bt, result=0;
            for(int j=0;j<kd;j++) {
                int idx=(h*kd+j)*vd+v;
                float s=state[idx]*decay+kk[j]*dv;
                state[idx]=s; result+=s*qq[j];
            }
            out[(t*vh+h)*vd+v]=b(result);
        }
        __syncthreads();
    }
}
extern "C" __global__ void gated_norm(const B* x, const B* gate, const float* w, B* out, int dim, float eps) {
    __shared__ float ss[256];
    int row=blockIdx.x,tid=threadIdx.x;
    float sum=0;
    for(int d=tid;d<dim;d+=256) { float v=f(x[row*dim+d]); sum+=v*v; }
    ss[tid]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) ss[tid]+=ss[tid+d]; __syncthreads(); }
    float inv=rsqrtf(ss[0]/dim+eps);
    for(int d=tid;d<dim;d+=256) out[row*dim+d]=b(f(b(f(b(f(x[row*dim+d])*inv))*w[d]))*silu(f(gate[row*dim+d])));
}
