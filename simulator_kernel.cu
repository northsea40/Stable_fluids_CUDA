//#pragma once
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "device_functions.h"
//#include<algorithm>
//
//__device__ unsigned IX(unsigned x, unsigned y, unsigned nw_)
//{
//	return y * nw_ + x;
//
//}
//__device__ float Clamp(float x, float min, float max)
//{
//	float ret = x;
//	if (x < min)ret = min;
//	if (x > max)ret = max;
//	return ret;
//}
//
//__device__ float Interpolation(float x, float y, float* u, unsigned nw_)
//{
//	int x_ceil = ceilf(x);
//	int x_floor = floorf(x);
//	int y_ceil = ceilf(y);
//	int y_floor = floorf(y);
//	int left_down_idx = IX(x_floor, y_floor, nw_);
//	int left_up_idx = IX(x_floor, y_ceil, nw_);
//	int right_down_idx = IX(x_ceil, y_floor, nw_);
//	int right_up_idx = IX(x_ceil, y_ceil, nw_);
//	float t1 = y - float(y_floor);
//	float t0 = 1 - t1;
//	float s1 = x - float(x_floor);
//	float s0 = 1 - s1;
//	return s0 * (t0 * u[left_down_idx] + t1 * u[left_up_idx]) + s1 * (t0 * u[right_down_idx] + t1 * u[right_up_idx]);
//}
//
//__global__ void AdvanceTime(float* oldarray, float* newarray, int nw_, int nh_)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid >= nw_ * nh_) return;
//	oldarray[tid] = newarray[tid];
//}
//
//__device__ void set_corner_gpu(float* x, int N)
//{
//	//x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
//	//x[IX(0, N - 1)] = 0.5 * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
//	//x[IX(N - 1, 0)] = 0.5 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
//	//x[IX(N - 1, N - 1)] = 0.5 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);
//}
//
//__device__ void  set_bnd_gpu(int b, float* x, int N, int tid)
//{
//	/*int i = tid % N;
//	int j = tid / N;
//	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1)
//	{
//
//		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
//		x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
//		x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
//		x[IX(N - 1, j)] = b == 1 ? -x[IX(N - 2, j)] : x[IX(N - 2, j)];
//	}
//	__syncthreads();
//
//	int n2 = N / 2;
//	if (i == n2 && j == n2)
//	{
//		set_corner_gpu(x, N);
//		return;
//	}*/
//}
//
//__device__ void  lin_solve_gpu(int b, float* x, float* x0, float a, float c, int iter, int N, int tid)
//{
//	//int localID = threadIdx.x;
//	//int i = tid % N;
//	//int j = tid / N;
//
//	//__shared__ float local_x[NUM_THREADS];
//
//	//if (i < 1 || i > N - 2) return;
//	//if (j < 1 || j > N - 2) return;
//
//	//float cRecip = 1.0 / c;
//	//for (int k = 0; k < iter; k++) 
//	//{
//	//	local_x[localID] =
//	//		(x0[IX(i, j)]
//	//			+ a * (x[IX(i + 1, j)]
//	//				+ x[IX(i - 1, j)]
//	//				+ x[IX(i, j + 1)]
//	//				+ x[IX(i, j - 1)]
//	//				)) * cRecip;
//
//	//	__syncthreads();
//
//	//	x[IX(i, j)] = local_x[localID];
//
//	//	//__syncthreads();
//
//	//	set_bnd_gpu(b, x, N, tid);
//
//	//}
//}
//
//__global__ void diffuse_gpu(int b, float* x, float* x0, float diff, float dt, int iter, int N)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid >= N * N) return;
//
//	float a = dt * diff * (N - 2) * (N - 2);
//	//lin_solve_gpu(b, x, x0, a, 1 + 6 * a, iter, N, tid);
//	float c = 1 + 6 * a;
//	float cRecip = 1.0 / c;
//	for (int k = 0; k < iter; k++) {
//		for (int i = 0; i < N * N; i++)
//		{
//			lin_solve_gpu(b, x, x0, a, 1 + 6 * a, iter, N, i);
//		}
//		for (int i = 0; i < N * N; i++)
//		{
//			set_bnd_gpu(b, x, N, i);
//
//		}
//	}
//
//}
//
//
//__global__ void project_gpu(float* velocX, float* velocY, float* p, float* div, int iter, int N)
//{
//	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	//if (tid >= N * N) return;
//
//	//int i = tid % N;
//	//int j = tid / N;
//
//	//if (i < 1 || i > N - 2) return;
//	//if (j < 1 || j > N - 2) return;
//
//
//	//div[IX(i, j)] = -0.5f * (
//	//	velocX[IX(i + 1, j)]
//	//	- velocX[IX(i - 1, j)]
//	//	+ velocY[IX(i, j + 1)]
//	//	- velocY[IX(i, j - 1)]
//	//	) / N;
//	//p[IX(i, j)] = 0;
//	////__syncthreads();
//
//	//set_bnd_gpu(0, div, N, tid);
//	////__syncthreads();
//
//	//set_bnd_gpu(0, p, N, tid);
//	////__syncthreads();
//
//	//lin_solve_gpu(0, p, div, 1, 6, iter, N, tid);
//	////__syncthreads();
//
//
//	//velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
//	//	- p[IX(i - 1, j)]) * N;
//	//velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
//	//	- p[IX(i, j - 1)]) * N;
//	////__syncthreads();
//
//	//set_bnd_gpu(1, velocX, N, tid);
//	////__syncthreads();
//
//	//set_bnd_gpu(2, velocY, N, tid);
//	////__syncthreads();
//
//}
//
//__global__ void advect_gpu_u(float* ux_, float* uy_, float* ux_next, float* uy_next, float timestep, int nw_, int nh_)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid >= nw_ * nh_) return;
//
//	int x = tid % nw_;
//	int y = tid / nw_;
//
//	if (x < 1 || y > nw_ - 2) return;
//	if (x < 1 || y > nh_ - 2) return;
//
//	/*float i0, i1, j0, j1;
//
//	float dtx = dt * (N - 2);
//	float dty = dt * (N - 2);
//
//	float s0, s1, t0, t1;
//	float tmp1, tmp2, x, y;
//
//	float Nfloat = N;
//	float ifloat = i;
//	float jfloat = j;
//
//	tmp1 = dtx * velocX[tid];
//	tmp2 = dty * velocY[tid];
//	x = ifloat - tmp1;
//	y = jfloat - tmp2;
//
//	if (x < 0.5f) x = 0.5f;
//	if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
//	i0 = floorf(x);
//	i1 = i0 + 1.0f;
//	if (y < 0.5f) y = 0.5f;
//	if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
//	j0 = floorf(y);
//	j1 = j0 + 1.0f;
//
//	s1 = x - i0;
//	s0 = 1.0f - s1;
//	t1 = y - j0;
//	t0 = 1.0f - t1;
//
//	int i0i = i0;
//	int i1i = i1;
//	int j0i = j0;
//	int j1i = j1;
//
//	d[tid] =
//		s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)])
//		+ s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);
//
//	__syncthreads();
//
//	set_bnd_gpu(b, d, N, tid);*/
//
//
//	float xPosPrev = x - timestep * ux_[IX(x, y, nw_)];
//	float yPosPrev = y - timestep * uy_[IX(x, y, nw_)];
//	xPosPrev = Clamp(xPosPrev, 0.0f, float(nw_));
//	yPosPrev = Clamp(yPosPrev, 0.0f, float(nh_));
//	//may be not ux_??
//	float ux_advected = Interpolation(xPosPrev, yPosPrev, ux_, nw_);
//	float uy_advected = Interpolation(xPosPrev, yPosPrev, uy_, nw_);
//	//test interpolatation	 
//	//add to ux uy
//	ux_next[IX(x, y, nw_)] = ux_advected;
//	uy_next[IX(x, y, nw_)] = uy_advected;
//	//ux_half[idx(x, y)] = ux_advected;
//	//uy_half[idx(x, y)] = uy_advected;
//
//}
//
//__global__ void advect_gpu_density(float* density_, float* density_next, float* ux_, float* uy_, float* ux_next, float* uy_next, float timestep, int nw_, int nh_)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid >= nw_ * nh_) return;
//
//	int x = tid % nw_;
//	int y = tid / nw_;
//
//	if (x < 1 || y > nw_ - 2) return;
//	if (x < 1 || y > nh_ - 2) return;
//
//	float xPosPrev = x - timestep * ux_[IX(x, y, nw_)];
//	float yPosPrev = y - timestep * uy_[IX(x, y, nw_)];
//	xPosPrev = Clamp(xPosPrev, 0.0f, float(nw_));
//	yPosPrev = Clamp(yPosPrev, 0.0f, float(nh_));
//	float density_advected = Interpolation(xPosPrev, yPosPrev, density_, nw_);
//	density_next[IX(x, y, nw_)] = density_advected;
//}