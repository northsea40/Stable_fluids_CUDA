#include <iostream>
#include <functional>
#include <chrono>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"
#include "fluid.hpp"

#include "scene.hpp"
#include <algorithm>


#define CHECK(call)                                                      \
{                                                                        \
   const cudaError_t error = call;                                       \
   if (error != cudaSuccess)                                             \
   {                                                                     \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                           \
   }                                                                     \
}


void processMouseInput(GLFWwindow* window, FluidScene* camera);

const int NUM_THREADS=32;
const int BLOCK_WIDTH = 64;
const int BLOCK_HEIGHT = 64;

__device__ unsigned IX(unsigned x, unsigned y, unsigned nw_)
{
	return y * nw_ + x;

}
__device__ double Clamp(double x, double min, double max)
{
	double ret = x;
	if (x < min)ret = min;
	if (x > max)ret = max;
	return ret;
}

//For advect part
__device__ double Interpolation(double x, double y, double* u, unsigned nw_)
{
	int x_ceil = ceilf(x);
	int x_floor = floorf(x);
	int y_ceil = ceilf(y);
	int y_floor = floorf(y);
	int left_down_idx = IX(x_floor, y_floor, nw_);
	int left_up_idx = IX(x_floor, y_ceil, nw_);
	int right_down_idx = IX(x_ceil, y_floor, nw_);
	int right_up_idx = IX(x_ceil, y_ceil, nw_);
	double t1 = y - double(y_floor);
	double t0 = 1 - t1;
	double s1 = x - double(x_floor);
	double s0 = 1 - s1;
	return s0 * (t0 * u[left_down_idx] + t1 * u[left_up_idx]) + s1 * (t0 * u[right_down_idx] + t1 * u[right_up_idx]);
}

//Set specific value for boundary
__device__ void Set_boundary(double* u,int x,int y,int nw_,int nh_)
{
	if (x == 0 && y == 0) 
	{
		u[IX(x, y,nw_)] = 0.25 * u[IX(1, 0,nw_)] + 0.25 * u[IX(0, 1,nw_)];
		return;
	}
	if (x == nw_-1 && y == 0)
	{
		u[IX(x, y, nw_)] = 0.25 * u[IX(nw_ - 2, 0, nw_)] + 0.25 * u[IX(nw_ - 1, 1, nw_)];
		return;
	}
	if (x == 0 && y == nh_ - 1)
	{
		u[IX(x, y, nw_)] = 0.25 * u[IX(0, nh_ - 2, nw_)] + 0.25 * u[IX(1, nh_ - 1, nw_)];
		return;
	}
	if (x == nw_ - 1 && y == nh_ - 1)
	{
		u[IX(x, y, nw_)] = 0.25 * u[IX(nw_ - 2, nh_ - 1, nw_)] + 0.25 * u[IX(nw_ - 1, nh_ - 2, nw_)];
		return;
	}
	if (x==0)
	{
		u[IX(x, y,nw_)] = 0.5f * u[IX(1, y,nw_)];
		return;
	}
	if (x == nw_ - 1)
	{
		u[IX(x, y, nw_)] = 0.5f * u[IX(nw_ - 2, y, nw_)];
		return;
	}
	if (y == 0)
	{
		u[IX(x, y, nw_)] = 0.5f * u[IX(x, 1, nw_)];
		return;
	}
	if (y == nh_ - 1)
	{
		u[IX(x, y, nw_)] = 0.5f * u[IX(x, nh_-2, nw_)];
		return;
	}
}
	

//For out of index fetch,use itself value
__device__ double Safe_fetch(double* u, int x, int y, int nw_, int nh_) 
{
	double safeIndex_x = Clamp(x, 0, nw_-1);
	double safeIndex_y = Clamp(y, 0, nh_-1);
	return u[IX(safeIndex_x, safeIndex_y, nw_)];
}


__device__ double Safe_fetch_shared(double* u_local,double* u_global ,int x_local, int y_local,int x_global,int y_global,int n_local, int n_global)
{
	if (x_local<0 || x_local>n_local - 1 || y_local<0 || y_local>n_local - 1) 
	{
		return Safe_fetch(u_global, x_global, y_global, n_global,n_global);
	}
	return u_local[IX(x_local, y_local, n_local)];
}


__global__ void AdvanceTime(double* oldarray, double* newarray, int nw_, int nh_)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	oldarray[IX(x, y, nw_)] = newarray[IX(x, y, nw_)];
}

//Set value to 0
__global__ void SetValue(double* input, double number, int nw_, int nh_) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	input[IX(x,y,nw_)] = 0;
}



__global__ void diffuse_gpu(double* ux_, double* ux_next,double viscosity , int nw_,int nh_)
{
#pragma region MyRegion


	//int x = threadIdx.x + blockIdx.x * blockDim.x;
	//int y = threadIdx.y + blockIdx.y * blockDim.y;
	//if (x >= nw_ || y >= nh_) return;


	//G-S AND R-B
	//if ((x + y) % 2 == 0) 
	//{
	//	double uxpre = Safe_fetch(ux_, x - 1, y, nw_, nh_);
	//	double uxnext = Safe_fetch(ux_, x + 1, y, nw_, nh_);
	//	double uypre = Safe_fetch(ux_, x, y - 1, nw_, nh_);
	//	double uynext = Safe_fetch(ux_, x, y + 1, nw_, nh_);
	//	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);
	//}
	//__syncthreads();
	//if ((x + y) % 2 != 0) 
	//{
	//	double uxpre = Safe_fetch(ux_next, x - 1, y, nw_, nh_);
	//	double uxnext = Safe_fetch(ux_next, x + 1, y, nw_, nh_);
	//	double uypre = Safe_fetch(ux_next, x, y - 1, nw_, nh_);
	//	double uynext = Safe_fetch(ux_next, x, y + 1, nw_, nh_);
	//	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);
	//}


	//Jac
//	double uxpre = Safe_fetch(ux_, x - 1, y, nw_, nh_);
//	double uxnext = Safe_fetch(ux_, x + 1, y, nw_, nh_);
//	double uypre = Safe_fetch(ux_, x, y - 1, nw_, nh_);
//	double uynext = Safe_fetch(ux_, x, y + 1, nw_, nh_);
//	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);
#pragma endregion



	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	__shared__ double ux_next_shared[(BLOCK_WIDTH + 2) * (BLOCK_HEIGHT + 2)];
	ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 1, BLOCK_WIDTH + 2)] = ux_next[IX(x, y, nw_)];
	if (threadIdx.x == 0)
	{
		ux_next_shared[IX(threadIdx.x, threadIdx.y + 1, BLOCK_WIDTH + 2)] = Safe_fetch(ux_next, x - 1, y, nw_, nh_);
	}
	if (threadIdx.x == blockDim.x - 1||x==nw_-1)
	{
		ux_next_shared[IX(threadIdx.x + 2, threadIdx.y + 1, BLOCK_WIDTH + 2)] = Safe_fetch(ux_next, x + 1, y, nw_, nh_);
	}
	if (threadIdx.y == 0)
	{
		ux_next_shared[IX(threadIdx.x + 1, threadIdx.y, BLOCK_WIDTH + 2)] = Safe_fetch(ux_next, x, y - 1, nw_, nh_);
	}
	if (threadIdx.y == blockDim.y - 1 || y == nw_ - 1)
	{
		ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 2, BLOCK_WIDTH + 2)] = Safe_fetch(ux_next, x, y + 1, nw_, nh_);
	}
	 // sync threads
	__syncthreads();
	double uxpre = ux_next_shared[IX(threadIdx.x, threadIdx.y + 1, BLOCK_WIDTH + 2)];
	double uxnext = ux_next_shared[IX(threadIdx.x + 2, threadIdx.y + 1, BLOCK_WIDTH + 2)];
	double uypre = ux_next_shared[IX(threadIdx.x + 1, threadIdx.y, BLOCK_WIDTH + 2)];
	double uynext = ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 2, BLOCK_WIDTH + 2)];
	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);


}


__global__ void diffuse_gpu_inner(double* ux_, double* ux_next, double viscosity, int nw_, int nh_) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	//allign with global index
	x += 1;
	y += 1;

	__shared__ double ux_next_shared[(BLOCK_WIDTH + 2) * (BLOCK_HEIGHT + 2)];
	ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 1, BLOCK_WIDTH + 2)] = ux_next[IX(x, y, nw_)];
	if (threadIdx.x == 0)
	{
		ux_next_shared[IX(threadIdx.x, threadIdx.y + 1, BLOCK_WIDTH + 2)] = ux_next[IX(x-1, y, nw_)];
	}
	if (threadIdx.x == blockDim.x - 1 || x == nw_ - 1)
	{
		ux_next_shared[IX(threadIdx.x + 2, threadIdx.y + 1, BLOCK_WIDTH + 2)] = ux_next[IX(x + 1, y, nw_)];
	}
	if (threadIdx.y == 0)
	{
		ux_next_shared[IX(threadIdx.x + 1, threadIdx.y, BLOCK_WIDTH + 2)] = ux_next[IX(x , y-1, nw_)];
	}
	if (threadIdx.y == blockDim.y - 1 || y == nw_ - 1)
	{
		ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 2, BLOCK_WIDTH + 2)] = ux_next[IX(x, y + 1, nw_)];
	}
	// sync threads
	__syncthreads();
	double uxpre = ux_next_shared[IX(threadIdx.x, threadIdx.y + 1, BLOCK_WIDTH + 2)];
	double uxnext = ux_next_shared[IX(threadIdx.x + 2, threadIdx.y + 1, BLOCK_WIDTH + 2)];
	double uypre = ux_next_shared[IX(threadIdx.x + 1, threadIdx.y, BLOCK_WIDTH + 2)];
	double uynext = ux_next_shared[IX(threadIdx.x + 1, threadIdx.y + 2, BLOCK_WIDTH + 2)];
	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);


}

__global__ void diffuse_gpu_border(double* ux_, double* ux_next, double viscosity, int nw_, int nh_) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	if (x > 0 || x < nw_ - 1 || y>0 || y < nh_ - 1) return;
	double uxpre = Safe_fetch(ux_next, x - 1, y, nw_,nh_);
	double uxnext = Safe_fetch(ux_next, x +1, y, nw_, nh_);
	double uypre = Safe_fetch(ux_next, x , y-1, nw_, nh_);
	double uynext = Safe_fetch(ux_next, x , y+1, nw_, nh_);
	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre + uxnext + uypre + uynext)) / (1.0f + 4.0f * viscosity);

}

__global__ void ComputePressure_gpu(double* ux_, double* uy_, double* pressure,double* pressure_next ,int nw_, int nh_) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	double diverg = 0.0f;
	double uxpre = Safe_fetch(ux_, x - 1, y, nw_, nh_);
	double uxnext = Safe_fetch(ux_, x + 1, y, nw_, nh_);
	double uypre = Safe_fetch(uy_, x , y-1, nw_, nh_);
	double uynext = Safe_fetch(uy_, x , y+1, nw_, nh_);
	diverg = -0.5f * (uxnext - uxpre + uynext - uypre);
	
	//G-S and R-B
	if ((x + y) % 2 == 0)
	{
		double pressurex_pre = Safe_fetch(pressure, x - 1, y, nw_, nh_);
		double pressurex_next = Safe_fetch(pressure, x + 1, y, nw_, nh_);
		double pressurey_pre = Safe_fetch(pressure, x, y + 1, nw_, nh_);
		double pressurey_next = Safe_fetch(pressure, x, y - 1, nw_, nh_);
		pressure_next[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
	}
	__syncthreads();
	if ((x + y) % 2 != 0)
	{
		double pressurex_pre = Safe_fetch(pressure_next, x - 1, y, nw_, nh_);
		double pressurex_next = Safe_fetch(pressure_next, x + 1, y, nw_, nh_);
		double pressurey_pre = Safe_fetch(pressure_next, x, y + 1, nw_, nh_);
		double pressurey_next = Safe_fetch(pressure_next, x, y - 1, nw_, nh_);
		pressure_next[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
	}


	//jac
	//double pressurex_pre = Safe_fetch(pressure, x - 1, y, nw_, nh_);
	//double pressurex_next = Safe_fetch(pressure, x + 1, y, nw_, nh_);
	//double pressurey_pre = Safe_fetch(pressure, x, y + 1, nw_, nh_);
	//double pressurey_next = Safe_fetch(pressure, x, y - 1, nw_, nh_);
	//pressure_next[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
}

__global__ void Projection_gpu(double* ux_, double* uy_, double* pressure, int nw_, int nh_) 
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	
	double pressurex_pre = Safe_fetch(pressure, x - 1, y, nw_, nh_);
	double pressurex_next = Safe_fetch(pressure, x + 1, y, nw_, nh_);
	double pressurey_pre = Safe_fetch(pressure, x , y-1, nw_, nh_);
	double pressurey_next = Safe_fetch(pressure, x, y+1, nw_, nh_);

	ux_[IX(x, y,nw_)] -= 0.5f * (pressurex_next - pressurex_pre);
	uy_[IX(x, y,nw_)] -= 0.5f * (pressurey_next - pressurey_pre);

}



__global__ void Advect_gpu_u(double* ux_, double* uy_, double* ux_next, double* uy_next, double* ux_half,double* uy_half ,double timestep, int nw_, int nh_)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y>=nh_) return;
	//TODO need check!
	//if (x < 1 || x > nw_ - 2)
	//{
	//	Set_boundary(ux_next, x, y, nw_, nh_);
	//	Set_boundary(uy_next, x, y, nw_, nh_);
	//	return;
	//}
	//if (y < 1 || y > nh_ - 2)
	//{
	//	Set_boundary(ux_next, x, y, nw_, nh_);
	//	Set_boundary(uy_next, x, y, nw_, nh_);
	//	return;
	//}
	double xPosPrev = x - timestep * ux_[IX(x, y, nw_)];
	double yPosPrev = y - timestep * uy_[IX(x, y, nw_)];

	//for for advanced part
	//double xPosPrev = x - 0.5f*timestep * ux_[IX(x, y, nw_)];
	//double yPosPrev = y - 0.5f*timestep * uy_[IX(x, y, nw_)];
	xPosPrev = Clamp(xPosPrev, 0.0f, double(nw_));
	yPosPrev = Clamp(yPosPrev, 0.0f, double(nh_));
	double ux_advected = Interpolation(xPosPrev, yPosPrev, ux_, nw_);
	double uy_advected = Interpolation(xPosPrev, yPosPrev, uy_, nw_);
	ux_next[IX(x, y, nw_)] = ux_advected;
	uy_next[IX(x, y, nw_)] = uy_advected;

	//not need to pay attention,for advanced part
	ux_half[IX(x, y, nw_)] = ux_advected;
	uy_half[IX(x, y, nw_)] = uy_advected;

}


//For advanced part need reflection
__global__ void Refelect_gpu_u(double* ux_, double* uy_, double* ux_half, double* uy_half,double timestep, int nw_, int nh_)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	ux_[IX(x, y, nw_)] = 2 * ux_[IX(x, y, nw_)] - ux_half[IX(x, y, nw_)];
	uy_[IX(x, y, nw_)] = 2 * uy_[IX(x, y, nw_)] - uy_half[IX(x, y, nw_)];

}


//density advection
__global__ void Advect_gpu_density(double* density_, double* density_next, double* ux_, double* uy_, double* ux_next, double* uy_next, double timestep, int nw_, int nh_)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	//if (x < 1 || x > nw_ - 2 || y < 1 || y > nh_ - 2)
	//{
	//	Set_boundary(density_next, x, y, nw_, nh_);
	//	return;
	//}

	double xPosPrev = x - timestep * ux_[IX(x, y, nw_)];
	double yPosPrev = y - timestep * uy_[IX(x, y, nw_)];
	xPosPrev = Clamp(xPosPrev, 0.0f, double(nw_));
	yPosPrev = Clamp(yPosPrev, 0.0f, double(nh_));
	double density_advected = Interpolation(xPosPrev, yPosPrev, density_, nw_);
	density_next[IX(x, y, nw_)] = density_advected;
}

__global__ void InletJetflow_gpu(double* ux_, double* density_, double t, int nw_, int nh_, int offset)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= 0 && x < 6 && y >= nh_ / 2 - 4 + offset && y < nh_ / 2 + 4 + offset)
	{
		//ux_[IX(x, y, nw_)] += t;
		//density_[IX(x, y, nw_)] = 1;
		//return;
	}
	if (x <= nw_ - 1 && x >= nw_ - 6 && y >= nh_ / 2 - 4 - offset && y < nh_ / 2 + 4 - offset)
	{
		ux_[IX(x, y, nw_)] -= t;
		density_[IX(x, y, nw_)] = 1;
		return;
	}
}

//For Debug
__global__ void Check_gpu(double* density_, double* density_next, double* ux_, double* uy_, double timestep, int nw_, int nh_)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	if (x < 6||y<6)
	{
		density_[IX(x, y, nw_)] = 1;
		return;
	}


}


//CPU part
#pragma region CPU Fuctions


unsigned idx(unsigned x, unsigned y, unsigned nw_)
{
	return y * nw_ + x;

}

void AdvanceTime_cpu(double* oldarray, double* newarray, int nw_, int nh_)
{
	for (size_t i = 0; i < nw_*nh_; i++)
	{
		oldarray[i] = newarray[i];
	}
}

double Clamp_cpu(double x, double min, double max)
{
	double ret = x;
	if (x < min)ret = min;
	if (x > max)ret = max;
	return ret;
}

double Interpolation_cpu(double x, double y, double* u, unsigned nw_)
{
	int x_ceil = ceilf(x);
	int x_floor = floorf(x);
	int y_ceil = ceilf(y);
	int y_floor = floorf(y);
	int left_down_idx = idx(x_floor, y_floor, nw_);
	int left_up_idx = idx(x_floor, y_ceil, nw_);
	int right_down_idx = idx(x_ceil, y_floor, nw_);
	int right_up_idx = idx(x_ceil, y_ceil, nw_);
	double t1 = y - double(y_floor);
	double t0 = 1 - t1;
	double s1 = x - double(x_floor);
	double s0 = 1 - s1;
	return s0 * (t0 * u[left_down_idx] + t1 * u[left_up_idx]) + s1 * (t0 * u[right_down_idx] + t1 * u[right_up_idx]);
}

void diffuse_cpu(double* ux_, double* ux_next, double viscosity, int nw_, int nh_)
{

	for (auto c = 0; c < 100; c++)
	{
		for (auto y = 1; y < nh_ - 1; y++)
		{
			for (auto x = 1; x < nw_ - 1; x++)
			{
				ux_next[idx(x, y, nw_)] = (ux_[idx(x, y, nw_)] + viscosity * (ux_next[idx(x - 1, y, nw_)] + ux_next[idx(x + 1, y, nw_)] + ux_next[idx(x, y - 1, nw_)] + ux_next[idx(x, y + 1, nw_)])) / (1.0f + 4.0f * viscosity);
			}
		}
	}

	//ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (ux_next[IX(x - 1, y, nw_)] + ux_next[IX(x + 1, y, nw_)] + ux_next[IX(x, y - 1, nw_)] + ux_next[IX(x, y + 1, nw_)])) / (1.0f + 4.0f * viscosity);
}

void advect_u_cpu(double* ux_,double* uy_ ,double* ux_next, double* uy_next,double timestep, int nw_, int nh_)
{

	
		for (auto y = 1; y < nh_ - 1; y++)
		{
			for (auto x = 1; x < nw_ - 1; x++)
			{
				double xPosPrev = x - timestep * ux_[idx(x, y, nw_)];
				double yPosPrev = y - timestep * uy_[idx(x, y, nw_)];
				xPosPrev = Clamp_cpu(xPosPrev, 0.0f, double(nw_));
				yPosPrev = Clamp_cpu(yPosPrev, 0.0f, double(nh_));
				//may be not ux_??
				double ux_advected = Interpolation_cpu(xPosPrev, yPosPrev, ux_, nw_);
				double uy_advected = Interpolation_cpu(xPosPrev, yPosPrev, uy_, nw_);
				//test interpolatation	 
				//add to ux uy
				ux_next[idx(x, y, nw_)] = ux_advected;
				uy_next[idx(x, y, nw_)] = uy_advected;
			}
		}

}

void advect_density_cpu(double* ux_, double* uy_, double* density_, double* density_next, double timestep , int nw_, int nh_)
{


	for (auto y = 1; y < nh_ - 1; y++)
	{
		for (auto x = 1; x < nw_ - 1; x++)
		{
			double xPosPrev = x - timestep * ux_[idx(x, y, nw_)];
			double yPosPrev = y - timestep * uy_[idx(x, y, nw_)];
			xPosPrev = Clamp_cpu(xPosPrev, 0.0f, double(nw_));
			yPosPrev = Clamp_cpu(yPosPrev, 0.0f, double(nh_));
			double density_advected = Interpolation_cpu(xPosPrev, yPosPrev, density_, nw_);
			density_next[idx(x, y, nw_)] = density_advected;
		}
	}

}

#pragma endregion



int main(int argc, char* argv[])
{
	//check cuda info

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

	printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);

	printf("Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("Warp size%d\n", deviceProp.warpSize);
	printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Maximum number of threads per multiprocessor: %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of warps per multiprocessor: %d\n",
		deviceProp.maxThreadsPerMultiProcessor / 32);



    GLFWwindow* window;
    unsigned width = 1000;
    unsigned height =1000 ;
	float totalTime = 0.0f;
	int loopTImes = 0;
    // Window setups
    {
        if (!glfwInit()) // Initialize glfw library
            return -1;

        // setting glfw window hints and global configurations
        {
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Use Core Mode
            // glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE); // Use Debug Context
        #ifdef __APPLE__
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
        #endif
        }

        // Create a windowed mode window and its OpenGL context
        window = glfwCreateWindow(width, height, "Stable Fluids Simulation", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return -1;
        }

        // window configurations
        {
            // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            // glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);
        }

        // Make the window's context current
        glfwMakeContextCurrent(window);

        // Load Opengl
        if (!gladLoadGL()) {
            glfwTerminate();
            return -1;
        };

        // On Window Resized
        glfwSetFramebufferSizeCallback(window,
            [](GLFWwindow*, int _w, int _h) {
                glViewport(0, 0, _w, _h);
            }
        );
    }

    // Main Loop
    {
        Shader shader("fluid.vs", "fluid.fs");

		//Set the resolution
        FluidScene fluid {1000, 1000};
        FluidVisualizer renderer {&shader, &fluid};

        int curKeyState = GLFW_RELEASE;
        int lastKeyState = GLFW_RELEASE;

		FluidData* GPU_Data = fluid.simulator_GPU->GPU_Data;
		FluidData* CPU_Data = fluid.simulator_GPU->CPU_Data;
		int nw = fluid.simulator_GPU->nw_;
		int nh = fluid.simulator_GPU->nh_;
		dim3 blks(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 grid((nw - 1) / blks.x + 1, (nh - 1) / blks.y + 1);

		dim3 blks_inner(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 grid_inner((nw-2 - 1) / blks.x + 1, (nh-2 - 1) / blks.y + 1);

		cudaError_t cudaStatus;
		int iterTime = 40;
		bool move_once = false;

        // Loop until the user closes the window
        while (!glfwWindowShouldClose(window))
        {
            // Terminate condition
            if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);

            processMouseInput(window, &fluid);
			            // Updating
            {
                lastKeyState = curKeyState;
                curKeyState = glfwGetKey(window, GLFW_KEY_P);
                // Uncomment to debug update step by step
               // if(curKeyState == GLFW_PRESS && lastKeyState == GLFW_RELEASE)
                if (curKeyState == GLFW_PRESS)
                {
					auto start = std::chrono::high_resolution_clock::now();
					#pragma region CPU Main Loop
					//fluid.step();
										// 
										// CPU CHECK
										//fluid.simulator_GPU->InletJetflow(CPU_Data,1);
										//advect_u_cpu(CPU_Data->ux_, CPU_Data->uy_, CPU_Data->ux_next ,CPU_Data->uy_next, fluid.simulator_GPU->timeStep,nw, nh);
										//AdvanceTime_cpu(CPU_Data->ux_, CPU_Data->ux_next, nw, nh);
										//AdvanceTime_cpu(CPU_Data->uy_, CPU_Data->uy_next, nw, nh);


										//diffuse_cpu(CPU_Data->ux_, CPU_Data->ux_next, fluid.simulator_GPU->viscosity, nw, nh);
										//diffuse_cpu(CPU_Data->uy_, CPU_Data->uy_next, fluid.simulator_GPU->viscosity, nw, nh);
										//AdvanceTime_cpu(CPU_Data->ux_, CPU_Data->ux_next, nw, nh);
										//AdvanceTime_cpu(CPU_Data->uy_, CPU_Data->uy_next, nw, nh);

										//advect_density_cpu(CPU_Data->ux_, CPU_Data->uy_, CPU_Data->density_, CPU_Data->density_next, fluid.simulator_GPU->timeStep, nw, nh);
										//AdvanceTime_cpu(CPU_Data->density_, CPU_Data->density_next, nw, nh);

										//diffuse_cpu(CPU_Data->density_, CPU_Data->density_next, fluid.simulator_GPU->diffk,nw, nh);
										//AdvanceTime_cpu(CPU_Data->density_, CPU_Data->density_next, nw, nh);
#pragma endregion

                    
					

					//GPU flow inject
					if (move_once == false)
					{
						fluid.simulator_GPU->CPU_TO_GPU(CPU_Data, GPU_Data);
						move_once = true;
					}
					InletJetflow_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->density_, 1, nw, nh, 5);
                   





			
					//diffuse u
					for (size_t i = 0; i < iterTime; i++)
					{
						diffuse_gpu <<<blks,grid >>> (GPU_Data->ux_, GPU_Data->ux_next, fluid.simulator_GPU->viscosity, nw, nh);
						cudaDeviceSynchronize();
						AdvanceTime << <blks, grid >> > (GPU_Data->ux_, GPU_Data->ux_next, nw, nh);
						cudaDeviceSynchronize();
						diffuse_gpu <<<blks,grid >>> (GPU_Data->uy_, GPU_Data->uy_next, fluid.simulator_GPU->viscosity, nw, nh);			
						cudaDeviceSynchronize();
						AdvanceTime << <blks, grid >> > (GPU_Data->uy_, GPU_Data->uy_next, nw, nh);
						cudaDeviceSynchronize();
					}
				
					//project u
					SetValue << <blks, grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					SetValue << <blks, grid >> > (GPU_Data->pressure_next, 0.0f, nw, nh);
					cudaDeviceSynchronize();
					for (size_t i = 0; i < iterTime; i++)
					{
						ComputePressure_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, GPU_Data->pressure_next, nw, nh);
						cudaDeviceSynchronize();
						AdvanceTime << <blks, grid >> > (GPU_Data->pressure, GPU_Data->pressure_next, nw, nh);
						cudaDeviceSynchronize();
					}
					Projection_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					cudaDeviceSynchronize();


					//advect u
					Advect_gpu_u << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, GPU_Data->ux_half, GPU_Data->uy_half, fluid.simulator_GPU->timeStep, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks, grid >> > (GPU_Data->ux_, GPU_Data->ux_next, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks, grid >> > (GPU_Data->uy_, GPU_Data->uy_next, nw, nh);
					cudaDeviceSynchronize();

					//project u
					SetValue << <blks,grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					SetValue << <blks, grid >> > (GPU_Data->pressure_next, 0.0f, nw, nh);
					cudaDeviceSynchronize();
					for (size_t i = 0; i < iterTime; i++)
					{
						ComputePressure_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, GPU_Data->pressure_next,nw, nh);
						cudaDeviceSynchronize();
						AdvanceTime << <blks, grid >> > (GPU_Data->pressure, GPU_Data->pressure_next, nw, nh);
						cudaDeviceSynchronize();
					}
					Projection_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					cudaDeviceSynchronize();

				

#pragma region Advanced part advection-reflection
					////reflect
					//Refelect_gpu_u << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_half, GPU_Data->uy_half, fluid.simulator_GPU->timeStep, nw, nh);
					//cudaDeviceSynchronize();

					////advect
					//Advect_gpu_u << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, GPU_Data->ux_half, GPU_Data->uy_half, fluid.simulator_GPU->timeStep, nw, nh);
					//cudaDeviceSynchronize();
					//AdvanceTime << <blks,grid >> > (GPU_Data->ux_, GPU_Data->ux_next, nw, nh);
					//cudaDeviceSynchronize();
					//AdvanceTime << <blks,grid >> > (GPU_Data->uy_, GPU_Data->uy_next, nw, nh);
					//cudaDeviceSynchronize();



					////projection
					//SetValue << <blks,grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					//cudaDeviceSynchronize();
					//for (size_t i = 0; i < iterTime; i++)
					//{
					//	ComputePressure_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					//	cudaDeviceSynchronize();
					//}
					//Projection_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					//cudaDeviceSynchronize();

					//
					////double projection
					//SetValue << <blks, grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					//cudaDeviceSynchronize();
					//for (size_t i = 0; i < iterTime; i++)
					//{
					//	ComputePressure_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					//	cudaDeviceSynchronize();
					//}
					//Projection_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					//cudaDeviceSynchronize();


#pragma endregion


					


			
					//diffuse density
					for (size_t i = 0; i < iterTime; i++)
					{
						diffuse_gpu << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, fluid.simulator_GPU->diffk, nw, nh);
						cudaDeviceSynchronize();
						AdvanceTime << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, nw, nh);
						cudaDeviceSynchronize();
					}
					
					//advect density
					Advect_gpu_density << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, fluid.simulator_GPU->timeStep, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, nw, nh);
					cudaDeviceSynchronize();



					auto end = std::chrono::high_resolution_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
					totalTime += duration.count();
					loopTImes += 1;


					//Check_gpu << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, fluid.simulator_GPU->timeStep, nw, nh);

					fluid.simulator_GPU->GPU_TO_CPU(CPU_Data, GPU_Data);
					//printf("doing\n");

					//CUDA BUG LOG
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
					}
                }
            }

            glClear(GL_COLOR_BUFFER_BIT);

            // Draw here
            {
                renderer.draw();
            }

            // Swap front and back buffers
            glfwSwapBuffers(window);

            // Poll for and process events
            glfwPollEvents();
        }
    }

    glfwTerminate();
	std::cout << "Execution time: " << totalTime/loopTImes << " microseconds" << std::endl;
    return 0;
}


void processMouseInput(GLFWwindow* window, FluidScene* fluid)
{
    static bool firstRun {true};
    static double lastCursorX {0};
    static double lastCursorY {0};
    static int lastLeftButtonState {GLFW_RELEASE};
    static int lastRightButtonState {GLFW_RELEASE};

    double curCursorX, curCursorY;
    int curLeftButtonState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    int curRightButtonState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

    glfwGetCursorPos(window, &curCursorX, &curCursorY);
    if (firstRun) {
        firstRun = false;
        lastCursorX = static_cast<double>(curCursorX);
        lastCursorY = static_cast<double>(curCursorY);
        return; // everything zero, so we return directly
    }

    if (curLeftButtonState == GLFW_PRESS || curRightButtonState == GLFW_PRESS) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        double deltaCursorX = (curCursorX - lastCursorX) / double(width);
        double deltaCursorY = (curCursorY - lastCursorY) / double(height);

        // map from screen to fluid space
        glm::vec2 applyPos = {
            curCursorX / width * fluid->width,
            (1.0 - curCursorY / height) * fluid->height
        };

        if (curLeftButtonState == GLFW_PRESS) {
            // Click to apply delta velocity
            glm::vec2 applyVel = {deltaCursorX * 100.0f, -deltaCursorY * 100.0f};
            fluid->applyImpulsiveVelocity(applyPos, applyVel);

            std::cout << "Apply ("<< applyPos.x << ", " << applyPos.y << "): (" << applyVel.x << ", " << applyVel.y << ")" << std::endl;

        } else if (lastRightButtonState == GLFW_RELEASE) {
            // Click to read data for debugging..?
            glm::vec2 vel = fluid->getVelocity(applyPos.x, applyPos.y); // interpolateVelocityAt(applyPos);
            double dens = fluid->getDensity(applyPos.x, applyPos.y); // interpolateDensityAt(applyPos);

            std::cout << "Read ("<< applyPos.x << ", " << applyPos.y << "): (" << vel.x << ", " << vel.y << ") " << dens << std::endl;
        }
    }

    // update record
    lastCursorX = static_cast<double>(curCursorX);
    lastCursorY = static_cast<double>(curCursorY);
    lastLeftButtonState = curLeftButtonState;
    lastRightButtonState = curRightButtonState;
}
