set(TARGET_NAME   gceed)
set(MODULE_LIB    mod)
set(CORE_LIB      core)


# Now, we support lapack only.
set(FILE_MATHLIB lapack)


### Build target
set(MAIN_SRC
    gceed.f90
    real_space_dft.f90
    calcELF.f90
    psl.f90
    hartree.f90
    xc.f90
    OUT_IN_data.f90
    subspace_diag.f90
    simple_mixing.f90
    copy_density.f90
    laplacianh.f90
    eigen_subdiag_${FILE_MATHLIB}.f90
    prep_ini.f90
    nabla.f90
    gram_schmidt.f90
    calc_rho_in.f90
    inner_product3.f90
    inner_product4.f90
    subdgemm_${FILE_MATHLIB}.f90
    calc_occupation.f90
    read_input_scf.f90
    init_wf.f90
    calc_force.f90
    sendrecv_copy.f90
    calc_gradient_fast.f90
    deallocate_sendrecv_groupob.f90
    calc_gradient_fast_c.f90
    check_numcpu.f90
    writepsi.f90
    calcJxyz.f90
    storevpp.f90
    calcuV.f90
    calcJxyz2nd.f90
    calcVpsl.f90
    bisection.f90
    structure_opt.f90
    setbN.f90
    setcN.f90
    ybcg.f90
    rmmdiis_eigen.f90
    rmmdiis.f90
    hartree_cg.f90
    hartree_boundary.f90
    calc_myob.f90
    check_corrkob.f90
    calc_allob.f90
    calc_pmax.f90
    calc_iroot.f90
    set_isstaend.f90
    conv_p0.f90
    conv_p.f90
    set_ispin.f90
    read_copy_pot.f90
    set_gridcoo.f90
    calc_dos.f90
    calc_pdos.f90
    real_time_dft.f90
    taylor.f90
    taylor_coe.f90
    WriteDensity.f90
    read_rt.f90
    time_evolution_step.f90
    hpsi_groupob.f90
    gradient_ex.f90
    calcEstatic.f90
    total_energy_groupob.f90
    xc_fast.f90
    calc_force_c.f90
    dip.f90
    calc_Mps3rd.f90
    read_input_rt.f90
    projection.f90
    add_polynomial.f90
    calcVbox.f90
    )

set(MODULE_SRC
    allocate_mat.f90
    new_world.f90
    init_sendrecv.f90
    sendrecv.f90
    laplacian2.f90
    gradient2.f90
    hpsi2.f90
    deallocate_mat.f90
    copy_psi_mesh.f90
    inner_product.f90
    share_mesh_1d_old.f90
    read_pslfile.f90
    total_energy.f90
    calc_density.f90
    change_order.f90
    allocate_sendrecv_groupob.f90
    allocate_psl.f90
    gradient.f90
    sendrecvh.f90
    calc_invA_${FILE_MATHLIB}.f90
    writebox_rt.f90
    readbox_rt.f90
    sendrecv_groupob.f90
    sendrecv_groupob_ngp.f90
    )

set(CORE_SRC
    calc_iquotient.f90
    calc_ob_num.f90
    read_input_gceed.f90
    rmmdiis_eigen_${FILE_MATHLIB}.f90
    scf_data.f90
    setlg.f90
    setmg.f90
    setng.f90
    ylm.f90
    copyVlocal.f90
    )

add_library(${CORE_LIB}    STATIC ${CORE_SRC})
add_library(${MODULE_LIB}  STATIC ${MODULE_SRC})
add_library(${TARGET_NAME} STATIC ${MAIN_SRC})

add_dependencies(${MODULE_LIB}  ${CORE_LIB})
add_dependencies(${TARGET_NAME} ${MODULE_LIB} ${CORE_LIB})

set(LIBRARIES
    ${MODULE_LIB}
    ${CORE_LIB}
    )

target_link_libraries(${TARGET_NAME} ${LIBRARIES})

list(APPEND SALMON_LINK_LIBRARIES ${TARGET_NAME})
set(SALMON_LINK_LIBRARIES ${SALMON_LINK_LIBRARIES} PARENT_SCOPE)
