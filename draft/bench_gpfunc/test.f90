program test
    implicit none

    integer :: i, j
    integer, parameter :: n1 = 10000, n2 = 32
    real(8), allocatable :: x(:, :)

    real(8) :: time_start, time_end, elapsed_time, time_min
    integer(8) :: count_start, count_end, count_rate, count_max, T, T_scale, k

    T = 1
    T_scale = 2
    time_min = 1d0

    allocate(x(n1, n2))
    x = 1.0d0

    do
    call system_clock(count_start, count_rate, count_max)
    do k = 1, T
    !$omp parallel
    !!$omp do schedule(static)
        do j = 1, n2
            !$omp do simd schedule(static)
            !!$omp simd
            do i = 1, n1
                x(i, j) = sin(exp(2.0d0*x(i, j)))
            end do
            !!$omp end simd
            !$omp end do simd
        end do
    !!$omp end do
    !$omp end parallel
    end do

    call system_clock(count_end, count_rate, count_max)

    time_start = (count_start*1.0_8)/count_rate
    time_end = (count_end*1.0_8)/count_rate
    elapsed_time = time_end - time_start

    if(elapsed_time < time_min) then
        T = int(T * T_scale)
    else
        print *, "         T (end) =", T
        print *, elapsed_time/real(T, 8)
        exit
    end if
    end do

    deallocate(x)

end program test
