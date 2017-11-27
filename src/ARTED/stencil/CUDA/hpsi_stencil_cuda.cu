#include <stdio.h>
#include <assert.h>
#include <cuComplex.h>

// 2 kernels, 1st kernel is optimized for reducing memory access as much as possible.

// #define USE_CONST  // use constant memory for arrays such as C() and D().

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

__host__ __device__ inline cuDoubleComplex cuCmul( double x, cuDoubleComplex y )
{
  cuDoubleComplex val;
  val = make_cuDoubleComplex ( x * cuCreal(y), x * cuCimag(y) );
  return val;
}

__host__ __device__ inline cuDoubleComplex cuCmulI( cuDoubleComplex x )
{
  cuDoubleComplex val;
  val = make_cuDoubleComplex ( - cuCimag(x), cuCreal(x) );
  return val;
}

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define DIV_CEIL(X,Y) (((X)+(Y)-1)/(Y))


/* */

#ifdef USE_CONST
#define MAX_NKB 512
__constant__ double  _C_const[12];
__constant__ double  _D_const[12*MAX_NKB];
#endif

/* */

static int is_init = 0;
static cudaStream_t st;

extern __shared__ void* _dyn_smem[];

/************************************************************/

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

  template<int NLX> // __launch_bounds__(128,2)
__global__ void hpsi1_rt_stencil_kern_v8_1(int Nkb,
    const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
    const cuDoubleComplex * __E, cuDoubleComplex *_F,
    int PNLx, int PNLy, int PNLz,
    int NLx, int NLy, int NLz,
    const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
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
  for (ix=0; ix < NLX; ix++) {
    val[ix] = make_cuDoubleComplex( 0.0, 0.0 );
  }

  int ixx;
#pragma unroll
  for (ix=0; ix < NLX; ix++) {
    cuDoubleComplex E_ix = E(iz,iy,ix);

    ixx = ix - 4; if (ixx < 0) ixx += NLX;
    val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(4),E_ix), cuCmulI(cuCmul(D(4),E_ix)) ) );
    if ( ix >= 8 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix - 3; if (ixx < 0) ixx += NLX;
    val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(3),E_ix), cuCmulI(cuCmul(D(3),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix - 2; if (ixx < 0) ixx += NLX;
    val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(2),E_ix), cuCmulI(cuCmul(D(2),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix - 1; if (ixx < 0) ixx += NLX;
    val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(1),E_ix), cuCmulI(cuCmul(D(1),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix;
    val[ixx] = cuCadd( val[ixx], cuCmul((B(iz,iy,ix)+A(ikb)), E_ix) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix + 1; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(1),E_ix), cuCmulI(cuCmul(D(1),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix + 2; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(2),E_ix), cuCmulI(cuCmul(D(2),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix + 3; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(3),E_ix), cuCmulI(cuCmul(D(3),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

    ixx = ix + 4; if (ixx >= NLX) ixx -= NLX;
    val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(4),E_ix), cuCmulI(cuCmul(D(4),E_ix)) ) );
    if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */
  }
}

  __launch_bounds__(128,4)
__global__ void hpsi1_rt_stencil_kern_v8_2(int Nkb,
    const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
    const cuDoubleComplex * __restrict__ __E, cuDoubleComplex *_F,
    int PNLx, int PNLy, int PNLz,
    int NLx, int NLy, int NLz,
    const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
  int ikb, ixz,iz,iy,ix;
  cuDoubleComplex v,w, val;

  ikb = blockIdx.y;
  if ( ikb >= Nkb ) return;

  const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

  ixz = threadIdx.x + blockDim.x * blockIdx.x;
  iz = ixz % NLz;
  ix = ixz / NLz;
  if ( ix >= NLx ) return;

  cuDoubleComplex E_idy[9];

  iy = 0;
  int  idy     = CALC_INDEX3(iz,PNLz,mody[iy+(-4)+NLy],PNLy,ix,PNLx);
  int  idy_min = CALC_INDEX3(0,PNLz,0  ,PNLy,ix,PNLx);
  int  idy_max = CALC_INDEX3(0,PNLz,NLy,PNLy,ix,PNLx);
  E_idy[0] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[1] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[2] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[3] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[4] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[5] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[6] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
  E_idy[7] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;

  for (iy=0; iy < NLy; iy++) {
    E_idy[8] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;

    v =         cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4))));
    w =         cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4))));
    v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
    w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
    v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
    w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
    v = cuCadd( cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1)))), v );
    w = cuCadd( cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1)))), w );

    v = cuCadd( cuCmul(C( 5),(cuCadd(E_idy[5],E_idy[3]))), v );
    w = cuCadd( cuCmul(D( 5),(cuCsub(E_idy[5],E_idy[3]))), w );
    v = cuCadd( cuCmul(C( 6),(cuCadd(E_idy[6],E_idy[2]))), v );
    w = cuCadd( cuCmul(D( 6),(cuCsub(E_idy[6],E_idy[2]))), w );
    v = cuCadd( cuCmul(C( 7),(cuCadd(E_idy[7],E_idy[1]))), v );
    w = cuCadd( cuCmul(D( 7),(cuCsub(E_idy[7],E_idy[1]))), w );
    v = cuCadd( cuCmul(C( 8),(cuCadd(E_idy[8],E_idy[0]))), v );
    w = cuCadd( cuCmul(D( 8),(cuCsub(E_idy[8],E_idy[0]))), w );

    val = cuCmul(-0.5, v);
    val = cuCsub( val, cuCmulI(w) );

    F(iz,iy,ix,ikb) = cuCadd(F(iz,iy,ix,ikb), val);

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

/************************************************************/

/*
 *
 */
void hpsi1_rt_stencil_gpu(double *_A,  // k2lap0_2(:)
    double *_B,  // Vloc
    double *_C,  // lapt(1:12)
    double *_D,  // nabt(1:12, ikb_s:ikb_e)
    cuDoubleComplex *_E,  //  tpsi(0:PNL-1, ikb_s:ikb_e)
    cuDoubleComplex *_F,  // htpsi(0:PNL-1, ikb_s:ikb_e)
    int IKB_s, int IKB_e, 
    int PNLx, int PNLy, int PNLz,
    int NLx, int NLy, int NLz,
    int *modx, int *mody, int *modz )
{
  if ( is_init == 0 ) {
    CUDA_CALL( cudaStreamCreate( &st ) );
    is_init = 1;
  }

  int Nkb = IKB_e - IKB_s + 1;

#ifdef USE_CONST
  CUDA_CALL( cudaMemcpyToSymbolAsync( _C_const, _C, sizeof(double)*12,     0, cudaMemcpyDeviceToDevice, st ) );

  assert( Nkb <= MAX_NKB );
  CUDA_CALL( cudaMemcpyToSymbolAsync( _D_const, _D, sizeof(double)*12*Nkb, 0, cudaMemcpyDeviceToDevice, st ) );
#endif

  {
    dim3 ts_1(128,1,1);
    dim3 bs_1(DIV_CEIL((NLy*NLz),ts_1.x),Nkb,1);
    if ( 0 ) {}
    else if ( NLx == 20 ) {
      hpsi1_rt_stencil_kern_v8_1<20><<< bs_1, ts_1, 0, st >>>( Nkb, _A, _B, _C, _D, _E, _F,
          PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
    }
    else if ( NLx == 16 ) {
      hpsi1_rt_stencil_kern_v8_1<16><<< bs_1, ts_1, 0, st >>>( Nkb, _A, _B, _C, _D, _E, _F,
          PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
    }
    else if ( NLx == 12 ) {
      hpsi1_rt_stencil_kern_v8_1<12><<< bs_1, ts_1, 0, st >>>( Nkb, _A, _B, _C, _D, _E, _F,
          PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
    }
    else { exit( -1 ); }

    dim3 ts_2(128,1,1);
    dim3 bs_2(DIV_CEIL((NLx*NLz),ts_2.x),Nkb,1);
    hpsi1_rt_stencil_kern_v8_2<<< bs_2, ts_2, 0, st >>>( Nkb, _A, _B, _C, _D, _E, _F,
        PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
    CUDA_CALL( cudaStreamSynchronize(st) );
  }
}

extern "C" {
  void hpsi1_rt_stencil_gpu_(double *A, double *B, double *C, double *D, cuDoubleComplex *E, cuDoubleComplex *F,
      int *IKB_s, int *IKB_e, 
      int *PNLx, int *PNLy, int *PNLz,
      int *NLx, int *NLy, int *NLz,
      int *modx, int *mody, int *modz ) {
    hpsi1_rt_stencil_gpu(A, B, C, D, E, F,
        *IKB_s, *IKB_e, 
        *PNLx, *PNLy, *PNLz,
        *NLx, *NLy, *NLz,
        modx, mody, modz );
  }
}
