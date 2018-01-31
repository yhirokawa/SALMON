#include <stdio.h>
#include <assert.h>
#include <cuComplex.h>

// 2 kernels, 1st kernel is optimized for reducing memory access as much as possible.

__host__ __device__ inline int CALC_INDEX2( int x, int nx, int y, int ny )
{
  return( x + nx * y );
}

__host__ __device__ inline int CALC_INDEX3( int x, int nx, int y, int ny, int z, int nz )
{
  return( CALC_INDEX2( CALC_INDEX2(x,nx,y,ny), nx*ny, z, nz ) );
}

__host__ __device__ inline int CALC_INDEX4( int x, int nx, int y, int ny, int z, int nz, int w, int nw )
{
  return( CALC_INDEX2( CALC_INDEX3(x,nx,y,ny,z,nz), nx*ny*nz, w, nw ) );
}



__host__ __device__ inline cuDoubleComplex conj_swap(cuDoubleComplex x)
{
  return make_cuDoubleComplex(-cuCimag(x), cuCreal(x));
}

__host__ __device__ inline cuDoubleComplex operator+=(cuDoubleComplex &x, cuDoubleComplex y)
{
  return x = cuCadd(x, y);
}

__host__ __device__ inline cuDoubleComplex operator-=(cuDoubleComplex &x, cuDoubleComplex y)
{
  return x = cuCsub(x, y);
}

__host__ __device__ inline cuDoubleComplex operator+(cuDoubleComplex x, cuDoubleComplex y)
{
  return cuCadd(x, y);
}

__host__ __device__ inline cuDoubleComplex operator-(cuDoubleComplex x, cuDoubleComplex y)
{
  return cuCsub(x, y);
}

__host__ __device__ inline cuDoubleComplex operator*(cuDoubleComplex x, cuDoubleComplex y)
{
  return cuCmul(x, y);
}

__host__ __device__ inline cuDoubleComplex operator*(double x, cuDoubleComplex y)
{
  return make_cuDoubleComplex(x * cuCreal(y), x * cuCimag(y));
}



#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define DIV_CEIL(X,Y) (((X)+(Y)-1)/(Y))


static int is_init = 0;
static cudaStream_t st;

/************************************************************/

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

template<int BSIZE>
__global__ __launch_bounds__(128,2)
void hpsi1_rt_stencil_ker1( int                                  Nkb
                          , const double          * __restrict__ _A
                          , const double          * __restrict__ _B
                          , const double          * __restrict__ _C
                          , const double          * __restrict__ _D
                          , const cuDoubleComplex * __restrict__ __E
                          ,       cuDoubleComplex *              _F
                          , int                                  PNLx
                          , int                                  PNLy
                          , int                                  PNLz
                          , int                                  NLx
                          , int                                  NLy
                          , int                                  NLz
                          )
{
  const int ikb = blockIdx.y;
  if(ikb < Nkb)
  {
    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    const int iyz = threadIdx.x + blockDim.x * blockIdx.x;
    const int iz  = iyz % NLz;
    const int iy  = iyz / NLz;
    if(iy < NLy)
    {
      for(int bx = 0; bx < NLx; bx += BSIZE)
      {
        cuDoubleComplex lmem[BSIZE];
        for(int ix = 0; ix < BSIZE; ++ix)
          lmem[ix] = make_cuDoubleComplex(0.0, 0.0);

#pragma unroll
        for(int ix = -4; ix < BSIZE+4; ++ix)
        {
          int idx = ix + bx;
          if     (idx <  0  ) idx += NLx;
          else if(idx >= NLx) idx -= NLx;
          const cuDoubleComplex E_idx = E(iz,iy,idx);

          int ixx;
          ixx = ix - 4; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(4) * E_idx) - conj_swap(D(4) * E_idx);
          ixx = ix - 3; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(3) * E_idx) - conj_swap(D(3) * E_idx);
          ixx = ix - 2; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(2) * E_idx) - conj_swap(D(2) * E_idx);
          ixx = ix - 1; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(1) * E_idx) - conj_swap(D(1) * E_idx);
          ixx = ix;     if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (A(ikb) + B(iz,iy,idx)) * E_idx;
          ixx = ix + 1; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(1) * E_idx) + conj_swap(D(1) * E_idx);
          ixx = ix + 2; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(2) * E_idx) + conj_swap(D(2) * E_idx);
          ixx = ix + 3; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(3) * E_idx) + conj_swap(D(3) * E_idx);
          ixx = ix + 4; if(0 <= ixx && ixx < BSIZE) lmem[ixx] = lmem[ixx] + (-0.5 * C(4) * E_idx) + conj_swap(D(4) * E_idx);
        }

#pragma unroll
        for(int ix = 0; ix < BSIZE; ++ix)
          if(ix+bx < NLx) F(iz,iy,ix+bx,ikb) = lmem[ix];
      }
    }
  }
}


template<int NLX> // __launch_bounds__(128,2)
__global__
void hpsi1_rt_stencil_ker1_c( int                                  Nkb
                            , const double          * __restrict__ _A
                            , const double          * __restrict__ _B
                            , const double          * __restrict__ _C
                            , const double          * __restrict__ _D
                            , const cuDoubleComplex * __restrict__ __E
                            ,       cuDoubleComplex *              _F
                            , int                                  PNLx
                            , int                                  PNLy
                            , int                                  PNLz
                            , int                                  NLx
                            , int                                  NLy
                            , int                                  NLz
                            )
{
  int ikb, iyz,iz,iy,ix;

  ikb = blockIdx.y;
  if ( ikb >= Nkb ) return;

  const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

  iyz = threadIdx.x + blockDim.x * blockIdx.x;
  iz = iyz % NLz;
  iy = iyz / NLz;
  if ( iy >= NLy ) return;

  cuDoubleComplex val[NLX];
  for (ix = 0; ix < NLX; ++ix)
    val[ix] = make_cuDoubleComplex(0.0, 0.0);

  int ixx;
#pragma unroll
  for (ix = 0; ix < NLX; ++ix) {
    cuDoubleComplex E_ix = E(iz,iy,ix);

    ixx = ix - 4; if (ixx < 0) ixx += NLX;
    val[ixx] = val[ixx] + (-0.5 * C(4) * E_ix) - conj_swap(D(4) * E_ix);
    if ( ix >= 8 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix - 3; if (ixx < 0) ixx += NLX;
    val[ixx] = val[ixx] + (-0.5 * C(3) * E_ix) - conj_swap(D(3) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix - 2; if (ixx < 0) ixx += NLX;
    val[ixx] = val[ixx] + (-0.5 * C(2) * E_ix) - conj_swap(D(2) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix - 1; if (ixx < 0) ixx += NLX;
    val[ixx] = val[ixx] + (-0.5 * C(1) * E_ix) - conj_swap(D(1) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix;
    val[ixx] = val[ixx] + (A(ikb) + B(iz,iy,ix)) * E_ix;
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix + 1; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = val[ixx] + (-0.5 * C(1) * E_ix) + conj_swap(D(1) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix + 2; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = val[ixx] + (-0.5 * C(2) * E_ix) + conj_swap(D(2) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix + 3; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = val[ixx] + (-0.5 * C(3) * E_ix) + conj_swap(D(3) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];

    ixx = ix + 4; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = val[ixx] + (-0.5 * C(4) * E_ix) + conj_swap(D(4) * E_ix);
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx];
  }
}

__global__ __launch_bounds__(128,4)
void hpsi1_rt_stencil_ker2( int                                  Nkb
                          , const double          * __restrict__ _C
                          , const double          * __restrict__ _D
                          , const cuDoubleComplex * __restrict__ __E
                          ,       cuDoubleComplex *              _F
                          , int                                  PNLx
                          , int                                  PNLy
                          , int                                  PNLz
                          , int                                  NLx
                          , int                                  NLy
                          , int                                  NLz
                          , const int             * __restrict__ modz
                          )
{
  const int ikb = blockIdx.y;
  if(ikb < Nkb)
  {
    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    const int ixz = threadIdx.x + blockDim.x * blockIdx.x;
    const int iz = ixz % NLz;
    const int ix = ixz / NLz;
    if(ix < NLx)
    {
      cuDoubleComplex E_idy[9];

      int iyy = NLy - 4;
      E_idy[0] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[1] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[2] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[3] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[4] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[5] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[6] = E(iz,iyy,ix); ++iyy; if(iyy >= NLy) iyy -= NLy;
      E_idy[7] = E(iz,iyy,ix);

      for(int iy = 0; iy < NLy ; ++iy)
      {
        iyy = iy + 4; if(iyy >= NLy) iyy -= NLy;
        E_idy[8] = E(iz,iyy,ix);

        cuDoubleComplex v, w;

        v  = C(12) * (E_IDZ(4) + E_IDZ(-4));
        w  = D(12) * (E_IDZ(4) - E_IDZ(-4));
        v += C(11) * (E_IDZ(3) + E_IDZ(-3));
        w += D(11) * (E_IDZ(3) - E_IDZ(-3));
        v += C(10) * (E_IDZ(2) + E_IDZ(-2));
        w += D(10) * (E_IDZ(2) - E_IDZ(-2));
        v += C( 9) * (E_IDZ(1) + E_IDZ(-1));
        w += D( 9) * (E_IDZ(1) - E_IDZ(-1));

        v += C( 5) * (E_idy[5] + E_idy[3]);
        w += D( 5) * (E_idy[5] - E_idy[3]);
        v += C( 6) * (E_idy[6] + E_idy[2]);
        w += D( 6) * (E_idy[6] - E_idy[2]);
        v += C( 7) * (E_idy[7] + E_idy[1]);
        w += D( 7) * (E_idy[7] - E_idy[1]);
        v += C( 8) * (E_idy[8] + E_idy[0]);
        w += D( 8) * (E_idy[8] - E_idy[0]);

        F(iz,iy,ix,ikb) += -0.5 * v - conj_swap(w);

        E_idy[0] = E_idy[1];
        E_idy[1] = E_idy[2];
        E_idy[2] = E_idy[3];
        E_idy[3] = E_idy[4];
        E_idy[4] = E_idy[5];
        E_idy[5] = E_idy[6];
        E_idy[6] = E_idy[7];
        E_idy[7] = E_idy[8];
      }
    }
  }
}

__global__ //__launch_bounds__(128,4)
void hpsi1_rt_stencil_full( int                                  Nkb
                          , const double          * __restrict__ _A
                          , const double          * __restrict__ _B
                          , const double          * __restrict__ _C
                          , const double          * __restrict__ _D
                          , const cuDoubleComplex * __restrict__ __E
                          ,       cuDoubleComplex *              _F
                          , int                                  PNLx
                          , int                                  PNLy
                          , int                                  PNLz
                          , int                                  NLx
                          , int                                  NLy
                          , int                                  NLz
                          , const int             * __restrict__ modx
                          , const int             * __restrict__ mody
                          , const int             * __restrict__ modz
                          )
{
  const int ikb = blockIdx.y;

  if(ikb < Nkb)
  {
    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    const int iyz = threadIdx.x + blockDim.x * blockIdx.x;
    const int iz = iyz % NLz;
    const int iy = iyz / NLz;

    if(iy < NLy)
    {
      for(int ix = 0; ix < NLx ; ++ix)
      {
        cuDoubleComplex v, w;

        v = C( 9) * (E_IDZ(1) + E_IDZ(-1))
          + C(10) * (E_IDZ(2) + E_IDZ(-2))
          + C(11) * (E_IDZ(3) + E_IDZ(-3))
          + C(12) * (E_IDZ(4) + E_IDZ(-4));
        w = D( 9) * (E_IDZ(1) - E_IDZ(-1))
          + D(10) * (E_IDZ(2) - E_IDZ(-2))
          + D(11) * (E_IDZ(3) - E_IDZ(-3))
          + D(12) * (E_IDZ(4) - E_IDZ(-4));

        v = C( 5) * (E_IDY(1) + E_IDY(-1))
          + C( 6) * (E_IDY(2) + E_IDY(-2))
          + C( 7) * (E_IDY(3) + E_IDY(-3))
          + C( 8) * (E_IDY(4) + E_IDY(-4)) + v;
        w = D( 5) * (E_IDY(1) - E_IDY(-1))
          + D( 6) * (E_IDY(2) - E_IDY(-2))
          + D( 7) * (E_IDY(3) - E_IDY(-3))
          + D( 8) * (E_IDY(4) - E_IDY(-4)) + w;

        v = C( 1) * (E_IDX(1) + E_IDX(-1))
          + C( 2) * (E_IDX(2) + E_IDX(-2))
          + C( 3) * (E_IDX(3) + E_IDX(-3))
          + C( 4) * (E_IDX(4) + E_IDX(-4)) + v;
        w = D( 1) * (E_IDX(1) - E_IDX(-1))
          + D( 2) * (E_IDX(2) - E_IDX(-2))
          + D( 3) * (E_IDX(3) - E_IDX(-3))
          + D( 4) * (E_IDX(4) - E_IDX(-4)) + w;

        F(iz,iy,ix,ikb) = (A(ikb) + B(iz,iy,ix)) * E(iz,iy,ix) - 0.5 * v - conj_swap(w);
      }
    }
  }
}

/*
 *
 */
void hpsi1_rt_stencil_gpu( double          *_A  // k2lap0_2(:)
                         , double          *_B  // Vloc
                         , double          *_C  // lapt(1:12)
                         , double          *_D  // nabt(1:12, ikb_s:ikb_e)
                         , cuDoubleComplex *_E  //  tpsi(0:PNL-1, ikb_s:ikb_e)
                         , cuDoubleComplex *_F  // htpsi(0:PNL-1, ikb_s:ikb_e)
                         , int              IKB_s
                         , int              IKB_e
                         , int              PNLx
                         , int              PNLy
                         , int              PNLz
                         , int              NLx
                         , int              NLy
                         , int              NLz
                         , int             *modx
                         , int             *mody
                         , int             *modz )
{
  if ( is_init == 0 ) {
    CUDA_CALL( cudaStreamCreate( &st ) );
    is_init = 1;
  }

  int Nkb = IKB_e - IKB_s + 1;

//#define USE_OPT
#ifdef USE_OPT
  dim3 t1(128);
  dim3 b1(DIV_CEIL((NLy*NLz),t1.x),Nkb);
  if (NLx == 20)
    hpsi1_rt_stencil_ker1_c<20><<<b1, t1, 0, st>>>(
      Nkb, _A, _B, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz
    );
  else if (NLx == 16)
    hpsi1_rt_stencil_ker1_c<16><<<b1, t1, 0, st>>>(
      Nkb, _A, _B, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz
    );
  else /* general */
    hpsi1_rt_stencil_ker1<8><<<b1, t1, 0, st>>>(
      Nkb, _A, _B, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz
    );

  dim3 t2(128);
  dim3 b2(DIV_CEIL((NLx*NLz),t2.x),Nkb);
  hpsi1_rt_stencil_ker2<<<b2, t2, 0, st>>>(
    Nkb, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz, modz
  );
#else
  dim3 t1(128);
  dim3 b1(DIV_CEIL((NLy*NLz),t1.x),Nkb);
  hpsi1_rt_stencil_full<<<b1, t1, 0, st>>>(
    Nkb, _A, _B, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz
  );
#endif

  CUDA_CALL(cudaStreamSynchronize(st));
}

extern "C" {
  void hpsi1_rt_stencil_gpu_( double *A
                            , double *B
                            , double *C
                            , double *D
                            , cuDoubleComplex *E
                            , cuDoubleComplex *F
                            , int *IKB_s
                            , int *IKB_e
                            , int *PNLx
                            , int *PNLy
                            , int *PNLz
                            , int *NLx
                            , int *NLy
                            , int *NLz
                            , int *modx
                            , int *mody
                            , int *modz )
  {
    hpsi1_rt_stencil_gpu( A, B, C, D, E, F
                        , *IKB_s, *IKB_e
                        , *PNLx, *PNLy, *PNLz
                        , *NLx, *NLy, *NLz
                        , modx, mody, modz );
  }

#define TEST_CUDA_SMEM_SIZE "TEST_CUDA_SMEM_SIZE"
  void set_cuda_l1cache_size_()
  {
    char *s = getenv(TEST_CUDA_SMEM_SIZE);
    if (s != NULL) {
      int v = atoi(s);
      printf("set smem size is %d%\n", v);
      CUDA_CALL(cudaFuncSetAttribute(hpsi1_rt_stencil_ker1_c<20>, cudaFuncAttributePreferredSharedMemoryCarveout, v));
      CUDA_CALL(cudaFuncSetAttribute(hpsi1_rt_stencil_ker1_c<16>, cudaFuncAttributePreferredSharedMemoryCarveout, v));
      CUDA_CALL(cudaFuncSetAttribute(hpsi1_rt_stencil_ker1<8>, cudaFuncAttributePreferredSharedMemoryCarveout, v));
      CUDA_CALL(cudaFuncSetAttribute(hpsi1_rt_stencil_ker2, cudaFuncAttributePreferredSharedMemoryCarveout, v));
      CUDA_CALL(cudaFuncSetAttribute(hpsi1_rt_stencil_full, cudaFuncAttributePreferredSharedMemoryCarveout, v));
    }
  }

  void init_cuda_()
  {
    //set_cuda_l1cache_size_();
  }
}
