!
!  Copyright 2017 SALMON developers
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!
module stencil_sub
complex(8), parameter :: zI=(0.d0,1.d0)

contains

!===================================================================================================================================

# define DX(dt) idx(ix+(dt)),iy,iz
# define DY(dt) ix,idy(iy+(dt)),iz
# define DZ(dt) ix,iy,idz(iz+(dt))

subroutine stencil_R(tpsi,htpsi,ipx_sta,ipx_end,ipy_sta,ipy_end,ipz_sta,ipz_end &
                    ,V_local,ix_sta,ix_end,iy_sta,iy_end,iz_sta,iz_end &
                    ,idx,idy,idz,lap0,lapt)
  implicit none
  integer,intent(in)  :: ipx_sta,ipx_end,ipy_sta,ipy_end,ipz_sta,ipz_end &
                        ,ix_sta,ix_end,iy_sta,iy_end,iz_sta,iz_end &
                        ,idx(ix_sta-4:ix_end+4),idy(iy_sta-4:iy_end+4),idz(iz_sta-4:iz_end+4)
  real(8),intent(in)  :: tpsi(ipx_sta:ipx_end,ipy_sta:ipy_end,ipz_sta:ipz_end) &
                        ,V_local(ix_sta:ix_end,iy_sta:iy_end,iz_sta:iz_end),lap0,lapt(4,3)
  real(8),intent(out) :: htpsi(ipx_sta:ipx_end,ipy_sta:ipy_end,ipz_sta:ipz_end)
  !
  integer :: iz,iy,ix
  real(8) :: v

!$OMP parallel
!$OMP do private(iz,iy,ix,v)
  do iz=iz_sta,iz_end
  do iy=iy_sta,iy_end
  do ix=ix_sta,ix_end

    v =  lapt(1,1)*(tpsi(DX(1)) + tpsi(DX(-1))) &
        +lapt(2,1)*(tpsi(DX(2)) + tpsi(DX(-2))) &
        +lapt(3,1)*(tpsi(DX(3)) + tpsi(DX(-3))) &
        +lapt(4,1)*(tpsi(DX(4)) + tpsi(DX(-4)))

    v =  lapt(1,2)*(tpsi(DY(1)) + tpsi(DY(-1))) &
        +lapt(2,2)*(tpsi(DY(2)) + tpsi(DY(-2))) &
        +lapt(3,2)*(tpsi(DY(3)) + tpsi(DY(-3))) &
        +lapt(4,2)*(tpsi(DY(4)) + tpsi(DY(-4))) + v

    v =  lapt(1,3)*(tpsi(DZ(1)) + tpsi(DZ(-1))) &
        +lapt(2,3)*(tpsi(DZ(2)) + tpsi(DZ(-2))) &
        +lapt(3,3)*(tpsi(DZ(3)) + tpsi(DZ(-3))) &
        +lapt(4,3)*(tpsi(DZ(4)) + tpsi(DZ(-4))) + v

    htpsi(ix,iy,iz) = ( V_local(ix,iy,iz) + lap0 )*tpsi(ix,iy,iz) - 0.5d0 * v
  end do
  end do
  end do
!$OMP end do
!$OMP end parallel

  return
end subroutine stencil_R

!===================================================================================================================================

subroutine stencil_C(tpsi,htpsi,ipx_sta,ipx_end,ipy_sta,ipy_end,ipz_sta,ipz_end &
                    ,V_local,ix_sta,ix_end,iy_sta,iy_end,iz_sta,iz_end &
                    ,idx,idy,idz,lap0,lapt,nabt)
  implicit none
  integer   ,intent(in)  :: ipx_sta,ipx_end,ipy_sta,ipy_end,ipz_sta,ipz_end &
                           ,ix_sta,ix_end,iy_sta,iy_end,iz_sta,iz_end &
                           ,idx(ix_sta-4:ix_end+4),idy(iy_sta-4:iy_end+4),idz(iz_sta-4:iz_end+4)
  complex(8),intent(in)  :: tpsi(ipx_sta:ipx_end,ipy_sta:ipy_end,ipz_sta:ipz_end)
  real(8)   ,intent(in)  :: V_local(ix_sta:ix_end,iy_sta:iy_end,iz_sta:iz_end),lap0,lapt(4,3),nabt(4,3)
  complex(8),intent(out) :: htpsi(ipx_sta:ipx_end,ipy_sta:ipy_end,ipz_sta:ipz_end)
  !
  integer :: iz,iy,ix
  complex(8) :: v,w
  complex(8) :: t(8)

!$OMP parallel do &
!$OMP          default(none) &
!$OMP          private(iz,iy,ix,v,w,t) &
!$OMP          firstprivate(iz_sta,iz_end,iy_sta,iy_end,ix_sta,ix_end,lap0) &
!$OMP          shared(lapt,nabt,idx,idy,idz,tpsi,V_local,htpsi)
  do iz=iz_sta,iz_end
  do iy=iy_sta,iy_end
!dir$ vector nontemporal(htpsi)
  do ix=ix_sta,ix_end

    t(1) = tpsi(DX( 1))
    t(2) = tpsi(DX( 2))
    t(3) = tpsi(DX( 3))
    t(4) = tpsi(DX( 4))
    t(5) = tpsi(DX(-1))
    t(6) = tpsi(DX(-2))
    t(7) = tpsi(DX(-3))
    t(8) = tpsi(DX(-4))

    v =  lapt(1,1)*(t(1)+t(5)) &
        +lapt(2,1)*(t(2)+t(6)) &
        +lapt(3,1)*(t(3)+t(7)) &
        +lapt(4,1)*(t(4)+t(8))

    w =  nabt(1,1)*(t(1)-t(5)) &
        +nabt(2,1)*(t(2)-t(6)) &
        +nabt(3,1)*(t(3)-t(7)) &
        +nabt(4,1)*(t(4)-t(8))

    t(1) = tpsi(DY( 1))
    t(2) = tpsi(DY( 2))
    t(3) = tpsi(DY( 3))
    t(4) = tpsi(DY( 4))
    t(5) = tpsi(DY(-1))
    t(6) = tpsi(DY(-2))
    t(7) = tpsi(DY(-3))
    t(8) = tpsi(DY(-4))

    v =  lapt(1,2)*(t(1)+t(5)) &
        +lapt(2,2)*(t(2)+t(6)) &
        +lapt(3,2)*(t(3)+t(7)) &
        +lapt(4,2)*(t(4)+t(8)) + v

    w =  nabt(1,2)*(t(1)-t(5)) &
        +nabt(2,2)*(t(2)-t(6)) &
        +nabt(3,2)*(t(3)-t(7)) &
        +nabt(4,2)*(t(4)-t(8)) + w

    t(1) = tpsi(DZ( 1))
    t(2) = tpsi(DZ( 2))
    t(3) = tpsi(DZ( 3))
    t(4) = tpsi(DZ( 4))
    t(5) = tpsi(DZ(-1))
    t(6) = tpsi(DZ(-2))
    t(7) = tpsi(DZ(-3))
    t(8) = tpsi(DZ(-4))

    v =  lapt(1,3)*(t(1)+t(5)) &
        +lapt(2,3)*(t(2)+t(6)) &
        +lapt(3,3)*(t(3)+t(7)) &
        +lapt(4,3)*(t(4)+t(8)) + v

    w =  nabt(1,3)*(t(1)-t(5)) &
        +nabt(2,3)*(t(2)-t(6)) &
        +nabt(3,3)*(t(3)-t(7)) &
        +nabt(4,3)*(t(4)-t(8)) + w

    htpsi(ix,iy,iz) = ( V_local(ix,iy,iz) + lap0 )*tpsi(ix,iy,iz) - 0.5d0 * v - zI * w
  end do
  end do
  end do
!$OMP end parallel do

  return
end subroutine stencil_C

!===================================================================================================================================

end module stencil_sub
