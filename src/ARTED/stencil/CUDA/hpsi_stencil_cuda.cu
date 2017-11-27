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

  dim3 t1(128);
  dim3 b1(DIV_CEIL((NLy*NLz),t1.x),Nkb);
  hpsi1_rt_stencil_ker1<8><<<b1, t1, 0, st>>>(
    Nkb, _A, _B, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz
  );

  dim3 t2(128);
  dim3 b2(DIV_CEIL((NLx*NLz),t2.x),Nkb);
  hpsi1_rt_stencil_ker2<<<b2, t2, 0, st>>>(
    Nkb, _C, _D, _E, _F, PNLx, PNLy, PNLz, NLx, NLy, NLz, modz
  );
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
}
