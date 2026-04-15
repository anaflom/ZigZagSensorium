subroutine assign_single_grid( grid,nintp, npart, pos, weight, dtl)
! Assigns particle weights to a 3D grid using various interpolation schemes (NGP, CIC, TSC, PCS).
! Arguments:
!   grid   - number of grid points per dimension
!   nintp  - interpolation order (1=NGP, 2=CIC, 3=TSC, 4=PCS)
!   npart  - number of particles
!   pos    - particle positions (3, npart)
!   weight - particle weights (npart)
!   dtl    - output grid (grid, grid, grid)

 integer, intent(in) ::  grid, nintp, npart
 real(kind=8), intent(in) :: pos(:,:),weight(:)
 real, parameter :: twopi = 6.28319
 real(kind=8), intent(out) :: dtl(grid, grid, grid)
 integer :: ix,iy,iz,ivr(3),iv(3,4),i,j,Nv(3), nintph
 real(kind=8) :: w(3,4),vr(3), h, h2
 
 if (nintp.ge.3) nintph=3   ! Set kernel half-width for TSC/PCS
 if (nintp.lt.3) nintph=1   ! Set kernel half-width for NGP/CIC
 
 Nv = (/grid,grid,grid/)    ! Grid size in each dimension
 w=0.d0                     ! Initialize weights
 
 do i=1,npart               ! Loop over all particles
 
   vr=dble(Nv)*pos(:,i)+1.d0   ! Compute grid coordinates (1-based)

   if (nintp.eq.4) then ! PCS (Piecewise Cubic Spline) interpolation
      ivr=int(vr)   ! Integer part of grid position
      do j=1,nintp
        iv(:,j) = mod(ivr(:)-nintph+j+Nv(:),Nv(:))+1   ! Neighboring grid indices (periodic BC)
      enddo
      do j=1,3
        h=vr(j)-int(vr(j))    ! Fractional part
        h2=h*h
        w(j,1)=(1.d0-h)**3/6.d0
        w(j,2)=4.d0/6.d0+(0.5d0*h-1.d0)*h2
        w(j,4)=h2*h/6.d0
        w(j,3)=1.d0-w(j,1)-w(j,2)-w(j,4)
      enddo
   endif   
   if (nintp.eq.3) then ! TSC (Triangular Shaped Cloud) interpolation
      ivr=nint(vr)
      do j=1,nintp
        iv(:,j) = mod(ivr(:)-nintph+j+Nv(:),Nv(:))+1   ! Neighboring grid indices (periodic BC)
      enddo
      do j=1,3
        h=vr(j)-nint(vr(j))   ! Fractional part
        h2=h*h
        w(j,1)=0.5d0*(0.5d0-h)**2
        w(j,2)=0.75d0-h2
        w(j,3)=1.d0-w(j,1)-w(j,2)
      enddo
   endif   
   if (nintp.eq.2) then ! CIC (Cloud In Cell) interpolation
      ivr=int(vr)
      do j=1,nintp
        iv(:,j) = mod(ivr(:)-nintph+j+Nv(:),Nv(:))+1   ! Neighboring grid indices (periodic BC)
      enddo
      do j=1,3
        h=vr(j)-nint(vr(j))   ! Fractional part
        w(j,1)=1.0d0-h
        w(j,2)=h
      enddo
   endif   
   if (nintp.eq.1) then ! NGP (Nearest Grid Point) interpolation
      w=1.0d0
      ivr=nint(vr)
      iv(:,1) = mod(ivr(:)-1+Nv(:),Nv(:))+1   ! Nearest grid index (periodic BC)
      
   endif
   
   ! Accumulate particle weight to grid points using computed weights
   do ix=1,nintp
      do iy=1,nintp
        do iz=1,nintp
          dtl(iv(1,ix),iv(2,iy),iv(3,iz))=dtl(iv(1,ix),iv(2,iy),iv(3,iz))+w(1,ix)*w(2,iy)*w(3,iz)*weight(i)
        enddo
      enddo
   enddo
 enddo
end subroutine

subroutine pcs_assign_2d(x_particles, y_particles, values, npart, Nx, Ny, xrange, yrange, grid)
! Assigns particle values to a 2D grid using the PCS (Piecewise Cubic Spline) kernel.
! Arguments:
!   x_particles, y_particles - particle positions (npart)
!   values                  - particle values (npart)
!   npart                   - number of particles
!   Nx, Ny                  - grid size in x and y
!   xrange, yrange          - physical range of grid in x and y (2-element arrays)
!   grid                    - output grid (Ny, Nx), accumulated in place

  integer, intent(in) :: npart, Nx, Ny
  real(8), intent(in) :: x_particles(:), y_particles(:), values(:)
  real(8), intent(in) :: xrange(2), yrange(2)
  real(8), intent(out) :: grid(Ny, Nx)

  integer :: idx, di, dj, ix, iy
  real(8) :: Dx, Dy
  real(8) :: xp, yp, val, ixc, iyc, sx, sy
  real(8) :: wx, wy
  real(8) :: abs_s

  grid(:,:) = 0.0d0   ! Initialize grid to zero

  Dx = (xrange(2) - xrange(1)) / real(Nx,8)   ! Grid spacing in x
  Dy = (yrange(2) - yrange(1)) / real(Ny,8)   ! Grid spacing in y

  do idx = 1, npart   ! Loop over all particles
   xp = x_particles(idx)
   yp = y_particles(idx)
   val = values(idx)
   ixc = (xp - xrange(1)) / Dx   ! Particle's grid coordinate (x)
   iyc = (yp - yrange(1)) / Dy   ! Particle's grid coordinate (y)

   do di = -2, 2 ! Loop over kernel support in x
     do dj = -2, 2 ! Loop over kernel support in y
      ix = int(floor(ixc + di))   ! Neighboring grid index (x)
      iy = int(floor(iyc + dj))   ! Neighboring grid index (y)
      if (ix >= 0 .and. ix < Nx .and. iy >= 0 .and. iy < Ny) then
        sx = ixc - real(ix,8)   ! Distance from grid point (x)
        sy = iyc - real(iy,8)   ! Distance from grid point (y)

        ! PCS kernel for sx
        abs_s = abs(sx)
        if (abs_s < 1.0d0) then
         wx = (1.0d0/6.0d0)*(4.0d0 - 6.0d0*abs_s**2 + 3.0d0*abs_s**3)
        else if (abs_s < 2.0d0) then
         wx = (1.0d0/6.0d0)*(2.0d0 - abs_s)**3
        else
         wx = 0.0d0
        end if

        ! PCS kernel for sy
        abs_s = abs(sy)
        if (abs_s < 1.0d0) then
         wy = (1.0d0/6.0d0)*(4.0d0 - 6.0d0*abs_s**2 + 3.0d0*abs_s**3)
        else if (abs_s < 2.0d0) then
         wy = (1.0d0/6.0d0)*(2.0d0 - abs_s)**3
        else
         wy = 0.0d0
        end if

        ! Note Fortran indices are 1-based; Python will need order='F' for easy mapping
        grid(iy+1, ix+1) = grid(iy+1, ix+1) + val*wx*wy   ! Accumulate weighted value
      end if
     end do
   end do
  end do
end subroutine pcs_assign_2d

subroutine pcs_assign_3d(x_particles, y_particles, z_particles, values, npart, Nx, Ny, Nz, xrange, yrange, zrange,grid)
! Assigns particle values to a 3D grid using the PCS (Piecewise Cubic Spline) kernel.
! Arguments:
!   x_particles, y_particles, z_particles - particle positions (npart)
!   values                               - particle values (npart)
!   npart                                - number of particles
!   Nx, Ny, Nz                           - grid size in x, y, z
!   xrange, yrange, zrange               - physical range of grid in x, y, z (2-element arrays)
!   grid                                 - output grid (Nz, Ny, Nx), accumulated in place

   integer, intent(in) :: npart, Nx, Ny, Nz
   real(8), intent(in) :: x_particles(:), y_particles(:), z_particles(:), values(:)
   real(8), intent(in) :: xrange(2), yrange(2), zrange(2)
   real(8), intent(out) :: grid(Nz, Ny, Nx)

   integer :: idx, di, dj, dk, ix, iy, iz
   real(8) :: Dx, Dy, Dz
   real(8) :: xp, yp, zp, val, ixc, iyc, izc, sx, sy, sz
   real(8) :: wx, wy, wz
   real(8) :: abs_s

   grid(:,:,:) = 0.0d0   ! Initialize grid to zero

   Dx = (xrange(2) - xrange(1)) / real(Nx,8)
   Dy = (yrange(2) - yrange(1)) / real(Ny,8)
   Dz = (zrange(2) - zrange(1)) / real(Nz,8)

   do idx = 1, npart
      xp = x_particles(idx)
      yp = y_particles(idx)
      zp = z_particles(idx)
      val = values(idx)
      ixc = (xp - xrange(1)) / Dx
      iyc = (yp - yrange(1)) / Dy
      izc = (zp - zrange(1)) / Dz

      do di = -2, 2
         ix = int(floor(ixc + di))
         if (ix < 0 .or. ix >= Nx) cycle
         sx = ixc - real(ix,8)
         abs_s = abs(sx)
         if (abs_s < 1.0d0) then
            wx = (1.0d0/6.0d0)*(4.0d0 - 6.0d0*abs_s**2 + 3.0d0*abs_s**3)
         else if (abs_s < 2.0d0) then
            wx = (1.0d0/6.0d0)*(2.0d0 - abs_s)**3
         else
            wx = 0.0d0
         end if

         do dj = -2, 2
            iy = int(floor(iyc + dj))
            if (iy < 0 .or. iy >= Ny) cycle
            sy = iyc - real(iy,8)
            abs_s = abs(sy)
            if (abs_s < 1.0d0) then
               wy = (1.0d0/6.0d0)*(4.0d0 - 6.0d0*abs_s**2 + 3.0d0*abs_s**3)
            else if (abs_s < 2.0d0) then
               wy = (1.0d0/6.0d0)*(2.0d0 - abs_s)**3
            else
               wy = 0.0d0
            end if

            do dk = -2, 2
               iz = int(floor(izc + dk))
               if (iz < 0 .or. iz >= Nz) cycle
               sz = izc - real(iz,8)
               abs_s = abs(sz)
               if (abs_s < 1.0d0) then
                  wz = (1.0d0/6.0d0)*(4.0d0 - 6.0d0*abs_s**2 + 3.0d0*abs_s**3)
               else if (abs_s < 2.0d0) then
                  wz = (1.0d0/6.0d0)*(2.0d0 - abs_s)**3
               else
                  wz = 0.0d0
               end if

               grid(iz+1, iy+1, ix+1) = grid(iz+1, iy+1, ix+1) + val*wx*wy*wz
            end do
         end do
      end do
   end do
end subroutine pcs_assign_3d
