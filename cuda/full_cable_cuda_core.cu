#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#define CUDA_OK(x) do{cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr float TEMP=34.f,PI=3.14159265358979323846f;
struct Param{float cm,ena,ek,gpas,epas,nax,na3,kdr,kdrb,kap,kad,kmb,kdb,hd,cal,can,cat,kca,cagk;};
struct Gate{float nam=0,nah=0,nas=1,kdr=0,kdrb=0,kapn=0,kapl=0,kadn=0,kadl=0,kmb=0,kdb=0,hd=0,cal=0,canm=0,canh=0,catm=0,cath=0,kca=0,cagk=0,cai=50e-6f;};
struct Rate{float inf,tau;};struct Spike{float time_ms;int32_t neuron;};
template<class T>std::vector<T>readb(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("read "+p);auto n=f.tellg();f.seekg(0);std::vector<T>v(size_t(n)/sizeof(T));f.read((char*)v.data(),n);return v;}
template<class T>void writeb(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));}
__device__ float trap(float v,float th,float a,float q){float x=v-th;return fabsf(x)>1e-6f?a*x/(1-expf(-x/q)):a*q;}
__device__ float adv(float x,float inf,float tau,float dt){return inf+(x-inf)*expf(-dt/fmaxf(1e-9f,tau));}
__device__ float efun(float z){return fabsf(z)<1e-4f?1-z/2:z/expm1f(z);}
__device__ float ghk(float v,float ci){float f=(25.f/293.15f)*(TEMP+273.15f)/2,nu=v/f;return-f*(1-(ci/2)*expf(nu))*efun(nu);}
__device__ Rate namr(float v){float qt=powf(2.f,(TEMP-24)/10),a=trap(v,-30,.4,7.2),b=trap(-v,30,.124,7.2);return{a/(a+b),fmaxf(.02,1/(a+b)/qt)};}
__device__ Rate nahr(float v){float qt=powf(2.f,(TEMP-24)/10),a=trap(v,-45,.03,1.5),b=trap(-v,45,.01,1.5);return{1/(1+expf((v+50)/4)),fmaxf(.5,1/(a+b)/qt)};}
__device__ Rate nasr(float v){float c=1/(1+expf((v+58)/2)),al=expf(1e-3f*12*(v+60)*9.648e4f/(8.315f*(273.16f+TEMP))),be=expf(1e-3f*12*.2f*(v+60)*9.648e4f/(8.315f*(273.16f+TEMP)));return{1.f,fmaxf(10.f,be/(.0003f*(1+al)))};}
__device__ Rate kdrate(float v,float vh,float a0,float z,float gm,float mn,float q10){float a=expf(1e-3f*z*(v-vh)*9.648e4f/(8.315f*(273.16f+TEMP))),b=expf(1e-3f*z*gm*(v-vh)*9.648e4f/(8.315f*(273.16f+TEMP))),qt=powf(q10,(TEMP-24)/10);return{1/(1+a),fmaxf(mn/qt,b/(qt*a0*(1+a)))};}
__device__ Rate kan(float v,bool dis){float z=(dis?-1.8f:-1.5f)-1/(1+expf((v+40)/5)),vh=dis?-1.f:11.f,a0=dis?.1f:.05f,gm=dis?.39f:.55f,mn=dis?.2f:.1f,a=expf(1e-3f*z*(v-vh)*9.648e4f/(8.315f*(273.16f+TEMP))),b=expf(1e-3f*z*gm*(v-vh)*9.648e4f/(8.315f*(273.16f+TEMP))),qt=powf(5.f,(TEMP-24)/10);return{1/(1+a),fmaxf(mn,b/(qt*a0*(1+a)))};}
__device__ Rate kal(float v){float a=expf(1e-3f*3*(v+56)*9.648e4f/(8.315f*(273.16f+TEMP)));return{1/(1+a),fmaxf(2.f,.26f*(v+50))};}
__device__ Rate kmbr(float v){float a=expf(.0378f*7*(v+42)),b=expf(.0378f*7*.4f*(v+42));return{1/(1+expf((v+40)/-10)),60+b/(.003f*(1+a))};}
__device__ Rate hdr(float v){float qt=powf(4.5f,(TEMP-33)/10),a=expf(.0378f*2.2f*(v+75)),b=expf(.0378f*2.2f*.4f*(v+75));return{1/(1+expf((v+81)/8)),b/(qt*.011f*(1+a))};}
__device__ Rate calr(float v){float z=(-v+81.5f)/10,a=fabsf(z)<1e-7f?156.9f:15.69f*(-v+81.5f)/expm1f(z),b=.29f*expf(-v/10.86f),am=expf(.0378f*2*(v-4)),bm=expf(.0378f*.2f*(v-4)),qt=powf(5.f,(TEMP-25)/10);return{a/(a+b),fmaxf(.2f/qt,bm/(qt*.1f*(1+am)))};}
__device__ Rate canmr(float v){float z=(-v+19.88f)/10,a=fabsf(z)<1e-7f?1.967f:.1967f*(-v+19.88f)/expm1f(z),b=.046f*expf(-v/20.73f),am=expf(.0378f*2*(v+14)),bm=expf(.0378f*.2f*(v+14)),qt=powf(5.f,(TEMP-25)/10);return{a/(a+b),fmaxf(.2f/qt,bm/(qt*.03f*(1+am)))};}
__device__ Rate canhr(float v){float a=1.6e-4f*expf(-v/48.4f),b=1/(expf((-v+39)/10)+1);return{a/(a+b),80};}
__device__ Rate catmr(float v){float z=(-v+19.26f)/10,a=fabsf(z)<1e-7f?2.f:.2f*(-v+19.26f)/expm1f(z),b=.009f*expf(-v/22.03f),am=expf(.0378f*2*(v+28)),bm=expf(.0378f*.2f*(v+28)),qt=powf(5.f,(TEMP-25)/10);return{a/(a+b),fmaxf(.2f,bm/(qt*.04f*(1+am)))};}
__device__ Rate cathr(float v){float a=1e-6f*expf(-v/16.26f),b=1/(expf((-v+29.79f)/10)+1),ah=expf(.0378f*3.5f*(v+75)),bh=expf(.0378f*3.5f*.6f*(v+75));return{a/(a+b),fmaxf(10.f,bh/(.015f*(1+ah)))};}
__device__ Rate kcar(float ca){float x=powf(ca/.00035f,4),ta=powf(3.f,(TEMP-22)/10);return{x/(1+x),fmaxf(.5f,1/.03f/(1+x)/ta)};}
__device__ Rate cagkr(float v,float ca){float F=96.485f,R=8.313424f,a=ca*.28f/(ca+.48e-3f*expf(-2*.84f*F*v/R/(273.15f+TEMP))),b=.48f/(1+ca/(.13e-6f*expf(-2*F*v/R/(273.15f+TEMP))));return{a/(a+b),1/(a+b)};}
__device__ void gate_step(Gate&q,float v,float dt,bool init){auto go=[&](float&x,Rate r){x=init?r.inf:adv(x,r.inf,r.tau,dt);};go(q.nam,namr(v));go(q.nah,nahr(v));go(q.nas,nasr(v));go(q.kdr,kdrate(v,13,.02,-3,.7,2,1));go(q.kdrb,kdrate(v,13,.02,-3,.7,2,1));go(q.kapn,kan(v,false));go(q.kapl,kal(v));go(q.kadn,kan(v,true));go(q.kadl,kal(v));go(q.kmb,kmbr(v));go(q.kdb,kdrate(v,-33,.005,3,.7,2,1));go(q.hd,hdr(v));go(q.cal,calr(v));go(q.canm,canmr(v));go(q.canh,canhr(v));go(q.catm,catmr(v));go(q.cath,cathr(v));go(q.kca,kcar(q.cai));go(q.cagk,cagkr(v,q.cai));}
__device__ float current(const Param&p,const Gate&q,float v,float*ica=nullptr){float I=p.gpas*(v-p.epas);I+=p.nax*q.nam*q.nam*q.nam*q.nah*(v-p.ena)+p.na3*q.nam*q.nam*q.nam*q.nah*q.nas*(v-p.ena);I+=(p.kdr*q.kdr+p.kdrb*q.kdrb+p.kap*q.kapn*q.kapl+p.kad*q.kadn*q.kadl+p.kmb*q.kmb+p.kdb*q.kdb+p.kca*q.kca*q.kca*q.kca+p.cagk*q.cagk)*(v-p.ek)+p.hd*q.hd*(v+30);float h2=.001f/(.001f+q.cai),ca=(p.cal*q.cal*q.cal+p.can*q.canm*q.canm*q.canh+p.cat*q.catm*q.catm*q.cath)*h2*ghk(v,q.cai);if(ica)*ica=ca;return I+ca;}
__device__ uint32_t rand32(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}__device__ float normal(uint32_t&x){float s=0;for(int i=0;i<6;i++)s+=(rand32(x)+1.f)*2.3283064e-10f;return(s-3)*1.41421356f;}

__global__ void full_cable(const uint8_t*ctype,const uint64_t*celloff,const uint64_t*toff,const int32_t*parent,const float*area,const float*gax,const Param*p,const float*rheo,int n,float dt,int steps,float warmup_ms,
                           float*V,Gate*Q,float*D,float*B,float*off,float*old,Spike*spikes,uint64_t capacity,unsigned long long*nsp,uint32_t*counts){
 int cell=blockIdx.x;if(cell>=n)return;int ty=ctype[cell];uint64_t base=celloff[cell],tb=toff[ty];int nc=int(toff[ty+1]-tb);__shared__ float inject,prev,ou;__shared__ uint32_t state;
 if(threadIdx.x==0){prev=-70;ou=0;state=20260822u^((cell+1)*0x9e3779b9u);counts[cell]=0;}for(int k=threadIdx.x;k<nc;k+=blockDim.x){V[base+k]=-70;Q[base+k]=Gate{};gate_step(Q[base+k],-70,dt,true);}__syncthreads();
 for(int step=0;step<steps;step++){
  if(threadIdx.x==0){ou+=dt/5.f*(-ou)+.35f*rheo[ty]*sqrtf(2*dt/5.f)*normal(state);inject=step*dt>=warmup_ms?fmaxf(0.f,1.30f*rheo[ty]+ou):0.f;}__syncthreads();
  for(int k=threadIdx.x;k<nc;k+=blockDim.x){uint64_t z=base+k;int q=tb+k;gate_step(Q[z],V[z],dt,false);old[z]=V[z];float eps=1e-3f,I=current(p[q],Q[z],V[z]),g=(current(p[q],Q[z],V[z]+eps)-current(p[q],Q[z],V[z]-eps))/(2*eps),G=g*area[q]*.01f,src=(g*V[z]-I)*area[q]*.01f,cap=p[q].cm*area[q]*1e-5f;D[z]=cap/dt+.5f*G;B[z]=(cap/dt-.5f*G)*V[z]+src;off[z]=0;}__syncthreads();
  for(int k=threadIdx.x+1;k<nc;k+=blockDim.x){uint64_t z=base+k,pa=base+parent[tb+k];float a=.5f*gax[tb+k];atomicAdd(&D[z],a);atomicAdd(&D[pa],a);atomicAdd(&B[z],.5f*gax[tb+k]*(old[pa]-old[z]));atomicAdd(&B[pa],.5f*gax[tb+k]*(old[z]-old[pa]));off[z]=-a;}__syncthreads();
  if(threadIdx.x==0){B[base]+=inject;for(int k=nc-1;k>0;k--){uint64_t z=base+k,pa=base+parent[tb+k];float f=off[z]/D[z];D[pa]-=off[z]*f;B[pa]-=B[z]*f;}V[base]=B[base]/D[base];for(int k=1;k<nc;k++){uint64_t z=base+k;V[z]=(B[z]-off[z]*V[base+parent[tb+k]])/D[z];}float now=(step+1)*dt;if(now>=warmup_ms&&V[base]>=0&&prev<0){counts[cell]++;unsigned long long slot=atomicAdd(nsp,1ULL);if(slot<capacity)spikes[slot]={now-warmup_ms,cell};}prev=V[base];}__syncthreads();
  for(int k=threadIdx.x;k<nc;k+=blockDim.x){uint64_t z=base+k;float ica=0;current(p[tb+k],Q[z],V[z],&ica);float drive=-ica/.1f/96485.33212f/2*1e4f,decay=.01f;Q[z].cai=fmaxf(1e-12f,(Q[z].cai+dt*(drive+50e-6f*decay))/(1+dt*decay));}__syncthreads();
 }
}

int main(int argc,char**argv){try{std::string in=argc>1?argv[1]:"mouse_ca1_network_gpu_test/inputs/full_morphology_templates_v1",types=argc>2?argv[2]:"mouse_ca1_connectome/outputs/full_connectome_39112_cuda_v1/input/cell_type_u8.bin",out=argc>3?argv[3]:"mouse_ca1_network_gpu_test/outputs/full_morphology_random_no_synapses_v1";float duration=argc>4?std::stof(argv[4]):100,dt=argc>5?std::stof(argv[5]):.025f,warmup=argc>6?std::stof(argv[6]):50.f;mkdir("mouse_ca1_network_gpu_test/outputs",0775);mkdir(out.c_str(),0775);
 auto ct=readb<uint8_t>(types);auto to=readb<uint64_t>(in+"/template_offsets_u64.bin");auto par=readb<int32_t>(in+"/parent_i32.bin");auto area=readb<float>(in+"/area_f32.bin"),gax=readb<float>(in+"/gax_f32.bin"),rheo=readb<float>(in+"/rheobase_f32.bin");std::vector<Param>hp(par.size());const char*names[]={"cm","ena","ek","gpas","epas","nax","na3","kdr","kdrb","kap","kad","kmb","kdb","hd","cal","can","cat","kca","cagk"};for(int f=0;f<19;f++){auto x=readb<float>(in+"/"+names[f]+"_f32.bin");for(size_t i=0;i<x.size();i++)reinterpret_cast<float*>(&hp[i])[f]=x[i];}
 int n=ct.size(),steps=roundf((duration+warmup)/dt);std::vector<uint64_t>co(n+1);for(int i=0;i<n;i++)co[i+1]=co[i]+to[ct[i]+1]-to[ct[i]];uint64_t total=co.back();std::cerr<<"cells="<<n<<" compartments="<<total<<" steps="<<steps<<" warmup_ms="<<warmup<<"\n";
 uint8_t*dct;uint64_t*dco,*dto;int32_t*dpar;float *da,*dg,*dr,*V,*D,*B,*of,*old;Param*dp;Gate*Q;uint32_t*dc;Spike*ds;unsigned long long*dn;constexpr uint64_t CAP=10000000;
 #define CP(ptr,v) CUDA_OK(cudaMalloc(&ptr,(v).size()*sizeof((v)[0])));CUDA_OK(cudaMemcpy(ptr,(v).data(),(v).size()*sizeof((v)[0]),cudaMemcpyHostToDevice))
 CP(dct,ct);CP(dco,co);CP(dto,to);CP(dpar,par);CP(da,area);CP(dg,gax);CP(dr,rheo);CP(dp,hp);CUDA_OK(cudaMalloc(&V,total*sizeof(float)));CUDA_OK(cudaMalloc(&Q,total*sizeof(Gate)));CUDA_OK(cudaMalloc(&D,total*sizeof(float)));CUDA_OK(cudaMalloc(&B,total*sizeof(float)));CUDA_OK(cudaMalloc(&of,total*sizeof(float)));CUDA_OK(cudaMalloc(&old,total*sizeof(float)));CUDA_OK(cudaMalloc(&dc,n*sizeof(uint32_t)));CUDA_OK(cudaMalloc(&ds,CAP*sizeof(Spike)));CUDA_OK(cudaMalloc(&dn,sizeof(unsigned long long)));CUDA_OK(cudaMemset(dn,0,sizeof(unsigned long long)));
 cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);cudaEventRecord(st);full_cable<<<n,128>>>(dct,dco,dto,dpar,da,dg,dp,dr,n,dt,steps,warmup,V,Q,D,B,of,old,ds,CAP,dn,dc);cudaEventRecord(en);CUDA_OK(cudaEventSynchronize(en));float ms;cudaEventElapsedTime(&ms,st,en);unsigned long long ns;CUDA_OK(cudaMemcpy(&ns,dn,sizeof(ns),cudaMemcpyDeviceToHost));if(ns>CAP)throw std::runtime_error("spike capacity");std::vector<Spike>sp(ns);std::vector<uint32_t>counts(n);CUDA_OK(cudaMemcpy(sp.data(),ds,ns*sizeof(Spike),cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(counts.data(),dc,n*sizeof(uint32_t),cudaMemcpyDeviceToHost));std::sort(sp.begin(),sp.end(),[](auto&a,auto&b){return a.time_ms<b.time_ms||(a.time_ms==b.time_ms&&a.neuron<b.neuron);});writeb(out+"/spikes_time_sorted.bin",sp);writeb(out+"/spike_count_by_neuron_u32.bin",counts);size_t active=0;for(auto x:counts)active+=x>0;
 std::ofstream j(out+"/simulation_report.json");j<<"{\n  \"backend\": \"CUDA C++\",\n  \"model\": \"Romani full morphology, 15 mechanisms, Hines tree Crank-Nicolson\",\n  \"synapses_enabled\": false,\n  \"cells\": "<<n<<",\n  \"instantiated_compartments\": "<<total<<",\n  \"recording_duration_ms\": "<<duration<<",\n  \"warmup_ms\": "<<warmup<<",\n  \"dt_ms\": "<<dt<<",\n  \"spikes\": "<<ns<<",\n  \"active_cells\": "<<active<<",\n  \"cuda_kernel_ms\": "<<ms<<",\n  \"morphology_scope\": \"12 type-specific full morphology/e-model templates replicated across cells for this test\"\n}\n";std::cerr<<"spikes="<<ns<<" active="<<active<<" kernel_ms="<<ms<<"\n";
 }catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}return 0;}
