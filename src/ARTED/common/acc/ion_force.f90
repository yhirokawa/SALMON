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
!--------10--------20--------30--------40--------50--------60--------70--------80--------90--------100-------110-------120-------130
subroutine Ion_Force_omp(Rion_update,GS_RT,ixy_m)
  use Global_Variables, only: zu_t,zu_m,zu_GS,NB,NBoccmax,calc_mode_gs,calc_mode_rt
  implicit none
  integer,intent(in) :: GS_RT
  logical,intent(in) :: Rion_update
  integer,intent(in),optional :: ixy_m

  select case(GS_RT)
    case(calc_mode_gs)
      call impl(Rion_update,zu_GS,NB)
    case(calc_mode_rt)
      if (present(ixy_m)) then
        call impl(Rion_update,zu_m(:,:,:,ixy_m),NBoccmax)
      else
        call impl(Rion_update,zu_t,NBoccmax)
      end if
    case default
      call err_finalize('ion_force_omp: gs_rt flag')
  end select

contains
  subroutine impl(Rion_update,zutmp,zu_NB)
    use Global_Variables
    use salmon_parallel, only: nproc_group_tdks, nproc_id_global
    use salmon_communication, only: comm_summation, comm_is_root, comm_sync_all
  use timer
    use salmon_math
    use projector
    implicit none
    logical,intent(in)       :: Rion_update
    integer,intent(in)       :: zu_NB
    complex(8),intent(inout) :: zutmp(NL,zu_NB,NK_s:NK_e)

    integer      :: ia,ib,ilma,ik,ix,iy,iz,n,j,i
    integer      :: ix_s,ix_e,iy_s,iy_e,iz_s,iz_e
    integer,allocatable:: idx(:),idy(:),idz(:)
    real(8)      :: rab(3),rab2,Gvec(3),G2,Gd
    real(8)      :: ftmp_l(3,NI),ftmp_l_kl(3,NI,NK_s:NK_e)
    real(8)      :: FionR(3,NI), FionG(3,NI),nabt_wrk(4,3)
    real(8)      :: pftmp(3),ftmp_l1(3,NI),ftmp_l2(3,NI)
    complex(8)   :: uVpsi,duVpsi(3)
    complex(8)   :: dzudr(3,NL,NB,NK_s:NK_e),dzudrzu(3),dzuekrdr(3)


    !flag_use_grad_wf_on_force is given in Gloval_Variable
    !if .true.   use gradient of wave-function (better accuracy)
    !if .false., old way which use gradient of potential is used (less accurate)

    call timer_begin(LOG_ION_FORCE)

    !ion-ion interaction (point charge Ewald)
    if (Rion_update) then

      !ion-ion: Ewald short interaction term (real space)
      ftmp_l=0.d0
!$omp parallel do private(ia, ik,ix,iy,iz,ib,rab,rab2) reduction(+:ftmp_l) collapse(5)
      do ia=1,NI
      do ix=-NEwald,NEwald
      do iy=-NEwald,NEwald
      do iz=-NEwald,NEwald
      do ib=1,NI
        ik=Kion(ia)
        if(ix**2+iy**2+iz**2 == 0 .and. ia == ib) cycle
        rab(1)=Rion(1,ia)-ix*aLx-Rion(1,ib)
        rab(2)=Rion(2,ia)-iy*aLy-Rion(2,ib)
        rab(3)=Rion(3,ia)-iz*aLz-Rion(3,ib)
        rab2=sum(rab(:)**2)
        ftmp_l(:,ia)=ftmp_l(:,ia)&
             &-Zps(Kion(ia))*Zps(Kion(ib))*rab(:)/sqrt(rab2)*(-erfc_salmon(sqrt(aEwald*rab2))/rab2&
             &-2*sqrt(aEwald/(rab2*Pi))*exp(-aEwald*rab2))
      end do
      end do
      end do
      end do
      end do
      FionR = ftmp_l


      !ion-ion: Ewald long interaction term (wave-number space)
      ftmp_l=0.d0
!$omp parallel private(ia) reduction(+:ftmp_l)
      do ia=1,NI
!$omp do private(ik,n,Gvec,G2,Gd)
      do n=NG_s,NG_e
        if(n == nGzero) cycle
        ik=Kion(ia)
        Gvec(1)=Gx(n); Gvec(2)=Gy(n); Gvec(3)=Gz(n)
        G2=sum(Gvec(:)**2)
        Gd=sum(Gvec(:)*Rion(:,ia))
        ftmp_l(:,ia) = ftmp_l(:,ia) &
        &   + Gvec(:)*(4*Pi/G2)*exp(-G2/(4*aEwald))*Zps(ik) &
        &     *zI*0.5d0*(conjg(rhoion_G(n))*exp(-zI*Gd)-rhoion_G(n)*exp(zI*Gd))
      end do
!$omp end do
      end do
!$omp end parallel

      call comm_summation(ftmp_l,FionG,3*NI,nproc_group_tdks)
      Fion = FionR + FionG

    end if


    call update_projector(kac)


    ! ion-electron 
    if(flag_use_grad_wf_on_force)then

    ! Use gradient of wave-func for calculating force on ions

    !(prepare gradient of w.f.)
    ix_s = 0  ;  ix_e = NLx-1
    iy_s = 0  ;  iy_e = NLy-1
    iz_s = 0  ;  iz_e = NLz-1

    allocate(idx(ix_s-4:ix_e+4),idy(iy_s-4:iy_e+4),idz(iz_s-4:iz_e+4))
    do i=ix_s-4,ix_e+4
      idx(i) = mod(NLx+i,NLx)
    end do
    do i=iy_s-4,iy_e+4
      idy(i) = mod(NLy+i,NLy)
    end do
    do i=iz_s-4,iz_e+4
      idz(i) = mod(NLz+i,NLz)
    end do

    nabt_wrk(1:4,1) = nabx(1:4)
    nabt_wrk(1:4,2) = naby(1:4)
    nabt_wrk(1:4,3) = nabz(1:4)


!$acc data pcopyin(vpsl_ia,occ,zutmp) pcopyout(ftmp_l1,ftmp_l2) pcreate(ftmp_l_kl,dzudr) &
!$acc      pcopyin(ekr_omp,kac,uv,mps,iuv,jxyz,a_tbl)

    call stencil_c_zu_acc(zutmp, dzudr, &
                          ix_s,ix_e, iy_s,iy_e, iz_s,iz_e, 1,zu_nb,nb, nk_s,nk_e, &
                          idx,idy,idz, nabt_wrk)

    !Force from Vlocal with wave-func gradient --
!$acc kernels
    ftmp_l_kl(:,:,:)= 0.d0
!$acc end kernels

!$acc parallel vector_length(128)
!$acc loop gang collapse(2) independent
    do ik=NK_s,NK_e
    do ib=1,NBoccmax

!$acc loop vector private(dzudrzu,pftmp) independent
    do i=1,NL

      dzudrzu(:) = conjg(dzudr(:,i,ib,ik))*zutmp(i,ib,ik)

      do ia=1,NI
        pftmp(:)= &
        &  (-2.d0)* dble(dzudrzu(:)*Vpsl_ia(i,ia)) &
        &         * occ(ib,ik)*Hxyz

        do j=1,3
!$acc atomic update
          ftmp_l_kl(j,ia,ik) = ftmp_l_kl(j,ia,ik) + pftmp(j)
!$acc end atomic
        end do
      end do

    enddo

    enddo
    enddo
!$acc end parallel

    call timer_begin(LOG_ALLREDUCE)
!$acc kernels
    ftmp_l1=0.d0
!$acc loop gang vector(128)
    do ik=NK_s,NK_e

      do ia=1,NI
        do j=1,3
!$acc atomic update
          ftmp_l1(j,ia) = ftmp_l1(j,ia) + ftmp_l_kl(j,ia,ik)
!$acc end atomic
        end do
      end do

    end do
!$acc end kernels
    call timer_end(LOG_ALLREDUCE)


    !Non-Local pseudopotential term using gradient of w.f.
!$acc kernels
    ftmp_l_kl(:,:,:)= 0.d0
!$acc end kernels

!$acc parallel vector_length(128)
!$acc loop gang collapse(2) independent
    do ik=NK_s,NK_e
    do ib=1,NBoccmax

!$acc loop vector private(uvpsi,duvpsi,dzuekrdr,pftmp) independent
       do ilma=1,Nlma
          ia=a_tbl(ilma)
           uVpsi   =0.d0
          duVpsi(:)=0.d0
          do j=1,Mps(ia)
             i=Jxyz(j,ia)
            uVpsi      =uVpsi + uV(j,ilma)*ekr_omp(j,ia,ik)*zutmp(i,ib,ik)
            dzuekrdr(:)=(dzudr(:,i,ib,ik)+zI*kAc(ik,:)*zutmp(i,ib,ik))*ekr_omp(j,ia,ik)
            duVpsi(:)  =duVpsi(:) + conjg(dzuekrdr(:))*uV(j,ilma)
          end do
          uVpsi    =uVpsi    *Hxyz
          duVpsi(:)=duVpsi(:)*Hxyz
          pftmp(:) = (-2d0)*dble(uVpsi*duVpsi(:))*iuV(ilma)*occ(ib,ik)

          do j=1,3
!$acc atomic update
            ftmp_l_kl(j,ia,ik) = ftmp_l_kl(j,ia,ik) + pftmp(j)
!$acc end atomic
          end do
       end do

    end do
    end do
!$acc end parallel

    call timer_begin(LOG_ALLREDUCE)
!$acc kernels
    ftmp_l2=0.d0
!$acc loop gang vector(128)
    do ik=NK_s,NK_e

      do ia=1,NI
        do j=1,3
!$acc atomic update
          ftmp_l2(j,ia) = ftmp_l2(j,ia) + ftmp_l_kl(j,ia,ik)
!$acc end atomic
        end do
      end do

    end do
!$acc end kernels
    call timer_end(LOG_ALLREDUCE)

!$acc end data

    call timer_begin(LOG_ALLREDUCE)
    call comm_summation(ftmp_l1,Floc,3*NI,nproc_group_tdks)
    call comm_summation(ftmp_l2,Fnl,3*NI,nproc_group_tdks)
    call timer_end(LOG_ALLREDUCE)


    else

    ! Use gradient of potential for calculating force on ions
    ! (older method, less accurate for larger grid size)

    call err_finalize('OpenACC version unsupports older method of ion_force')

    endif ! flag_use_grad_wf_on_force


    call timer_begin(LOG_ALLREDUCE)
    force = Fion + Floc + Fnl
    call timer_end(LOG_ALLREDUCE)

    call timer_end(LOG_ION_FORCE)
  end subroutine

!Gradient of wave function (du/dr) with nine points formura
# define DX(dt) iz,iy,idx(ix+(dt)),ib,ik
# define DY(dt) iz,idy(iy+(dt)),ix,ib,ik
# define DZ(dt) idz(iz+(dt)),iy,ix,ib,ik
  subroutine stencil_C_zu_acc(zu0,dzu0dr &
  &                          ,ix_s,ix_e,iy_s,iy_e,iz_s,iz_e,ib_s,ib_e,ib_e2,ik_s,ik_e &
  &                          ,idx,idy,idz,nabt)
  implicit none
  integer   ,intent(in)  :: ix_s,ix_e,iy_s,iy_e,iz_s,iz_e,ib_s,ib_e,ib_e2,ik_s,ik_e
  integer   ,intent(in)  :: idx(ix_s-4:ix_e+4),idy(iy_s-4:iy_e+4),idz(iz_s-4:iz_e+4)
  real(8)   ,intent(in)  :: nabt(4,3)
  complex(8),intent(in)  :: zu0(iz_s:iz_e,iy_s:iy_e,ix_s:ix_e,ib_s:ib_e,ik_s:ik_e)
  complex(8),intent(out) :: dzu0dr(3,iz_s:iz_e,iy_s:iy_e,ix_s:ix_e,ib_s:ib_e2,ik_s:ik_e)
  !
  integer :: iz,iy,ix,ib,ik
  complex(8) :: w(3)

!$acc data pcopyin(zu0) pcopyout(dzu0dr)

!$acc kernels
!$acc loop gang collapse(2)
  do ik=ik_s,ik_e
  do ib=ib_s,ib_e

!$acc loop vector(128) collapse(2) private(w)
  do ix=ix_s,ix_e
  do iy=iy_s,iy_e
  do iz=iz_s,iz_e

    w(1) =  nabt(1,1)*(zu0(DX(1)) - zu0(DX(-1))) &
           +nabt(2,1)*(zu0(DX(2)) - zu0(DX(-2))) &
           +nabt(3,1)*(zu0(DX(3)) - zu0(DX(-3))) &
           +nabt(4,1)*(zu0(DX(4)) - zu0(DX(-4)))

    w(2) =  nabt(1,2)*(zu0(DY(1)) - zu0(DY(-1))) &
           +nabt(2,2)*(zu0(DY(2)) - zu0(DY(-2))) &
           +nabt(3,2)*(zu0(DY(3)) - zu0(DY(-3))) &
           +nabt(4,2)*(zu0(DY(4)) - zu0(DY(-4)))

    w(3) =  nabt(1,3)*(zu0(DZ(1)) - zu0(DZ(-1))) &
           +nabt(2,3)*(zu0(DZ(2)) - zu0(DZ(-2))) &
           +nabt(3,3)*(zu0(DZ(3)) - zu0(DZ(-3))) &
           +nabt(4,3)*(zu0(DZ(4)) - zu0(DZ(-4)))

    dzu0dr(:,iz,iy,ix,ib,ik) = w(:)
  end do
  end do
  end do

  end do
  end do
!$acc end kernels

!$acc end data

  end subroutine

end subroutine Ion_Force_omp
!--------10--------20--------30--------40--------50--------60--------70--------80--------90--------100-------110-------120-------130
