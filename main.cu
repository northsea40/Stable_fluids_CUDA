#include <iostream>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"
#include "fluid.hpp"

#include "scene.hpp"
#include <algorithm>

void processMouseInput(GLFWwindow* window, FluidScene* camera);

const int NUM_THREADS=32;

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
	

__global__ void AdvanceTime(double* oldarray, double* newarray, int nw_, int nh_)
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nw_ * nh_) return;*/
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	oldarray[IX(x, y, nw_)] = newarray[IX(x, y, nw_)];
}

__global__ void SetValue(double* input, double number, int nw_, int nh_) 
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nw_ * nh_) return;*/

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;
	input[IX(x,y,nw_)] = 0;
}



__device__ void  lin_solve_gpu(int b, double* x, double* x0, double a, double c, int iter, int N, int tid)
{
	//int localID = threadIdx.x;
	//int i = tid % N;
	//int j = tid / N;

	//__shared__ double local_x[NUM_THREADS];

	//if (i < 1 || i > N - 2) return;
	//if (j < 1 || j > N - 2) return;

	//double cRecip = 1.0 / c;
	//for (int k = 0; k < iter; k++) 
	//{
	//	local_x[localID] =
	//		(x0[IX(i, j)]
	//			+ a * (x[IX(i + 1, j)]
	//				+ x[IX(i - 1, j)]
	//				+ x[IX(i, j + 1)]
	//				+ x[IX(i, j - 1)]
	//				)) * cRecip;

	//	__syncthreads();

	//	x[IX(i, j)] = local_x[localID];

	//	//__syncthreads();

	//	set_bnd_gpu(b, x, N, tid);

	//}
}

__global__ void diffuse_gpu(double* ux_, double* ux_next,double viscosity , int nw_,int nh_)
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int localid = threadIdx.x;
	if (tid >= nw_ * nh_) return;
	int x = tid % nw_;
	int y = tid / nw_;*/

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	//if (x < 1 || x > nw_ - 2) 
	//{
	//	Set_boundary(ux_next, x, y, nw_, nh_);
	//	return;
	//} 
	//if (y < 1 || y > nh_ - 2) 
	//{
	//	Set_boundary(ux_next, x, y, nw_, nh_);
	//	return;
	//}
	/*if (IX(x-1, y, nw_) < 0 || IX(x-1, y, nw_) > 64 * 64 - 1) 
	{
		printf("fail:%d \n", IX(x - 1, y, nw_));
		printf("x:%d \n", x);
		printf("y:%d \n", y);
	}
	if (IX(x+1, y, nw_) < 0 || IX(x+1, y, nw_) > 64 * 64 - 1)
	{
		printf("fail:%d \n", IX(x + 1, y, nw_));
		printf("x:%d \n", x);
		printf("y:%d \n", y);
	}
	if (IX(x, y-1, nw_) < 0 || IX(x, y-1, nw_) > 64 * 64 - 1)
	{
		printf("fail:%d \n", IX(x, y-1, nw_));
		printf("x:%d \n", x);
		printf("y:%d \n", y);
	}
	if (IX(x, y+1, nw_) < 0 || IX(x, y+1, nw_) > 64 * 64 - 1)
	{
		printf("fail:%d \n", IX(x, y+1, nw_));
		printf("x:%d \n", x);
		printf("y:%d \n", y);
	}*/
	double uxpre;
	double uypre;
	double uxnext;
	double uynext;
	if (x < 1)
	{
		uxpre = ux_next[IX(x, y, nw_)];
	}
	else
	{
		uxpre = ux_next[IX(x - 1, y, nw_)];
	}

	if (y < 1)
	{
		uypre = ux_next[IX(x, y, nw_)];
	}
	else
	{
		uypre = ux_next[IX(x, y - 1, nw_)];
	}

	if (x > nh_ - 2)
	{
		uxnext = ux_next[IX(x, y, nw_)];
	}
	else
	{
		uxnext = ux_next[IX(x + 1, y, nw_)];
	}
	if (y > nh_ - 2)
	{
		uynext = ux_next[IX(x, y, nw_)];
	}
	else
	{
		uynext = ux_next[IX(x, y + 1, nw_)];
	}


	ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (uxpre+ uxnext + uypre+ uynext)) / (1.0f + 4.0f * viscosity);
	
	/*for (size_t i = 0; i < 100; i++)
	{
		if ((x + y) % 2 == 0) 
		{
			ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (ux_next[IX(x - 1, y, nw_)] + ux_next[IX(x + 1, y, nw_)] + ux_next[IX(x, y - 1, nw_)] + ux_next[IX(x, y + 1, nw_)])) / (1.0f + 4.0f * viscosity);
		}
		__syncthreads();
		if ((x + y) % 2 != 0)
		{
			ux_next[IX(x, y, nw_)] = (ux_[IX(x, y, nw_)] + viscosity * (ux_next[IX(x - 1, y, nw_)] + ux_next[IX(x + 1, y, nw_)] + ux_next[IX(x, y - 1, nw_)] + ux_next[IX(x, y + 1, nw_)])) / (1.0f + 4.0f * viscosity);
		}
	}*/
	
}


__global__ void ComputePressure_gpu(double* ux_, double* uy_, double* pressure, int nw_, int nh_) 
{
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//if (tid >= nw_ * nh_) return;

	//int x = tid % nw_;
	//int y = tid / nw_;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	/*if (x < 1 || x > nw_ - 2)
	{
		Set_boundary(ux_, x, y, nw_, nh_);
		return;
	}
	if (y < 1 || y > nh_ - 2)
	{
		Set_boundary(ux_, x, y, nw_, nh_);
		return;
	}*/
	double diverg = 0.0f;
	//__shared__ double pressure[NUM_THREADS];
	/*double ux_pre = ux_[IX(x - 1, y, nw_)];
	double uy_pre = uy_[IX(x, y - 1, nw_)];
	double ux_next = ux_[IX(x + 1, y, nw_)];
	double uy_next = uy_[IX(x, y + 1, nw_)];*/

	double uxpre;
	double uypre;
	double uxnext;
	double uynext;
	if (x < 1)
	{
		uxpre = ux_[IX(x, y, nw_)];
	}
	else
	{
		uxpre = ux_[IX(x - 1, y, nw_)];
	}

	if (y < 1)
	{
		uypre = uy_[IX(x, y, nw_)];
	}
	else
	{
		uypre = uy_[IX(x, y - 1, nw_)];
	}

	if (x > nh_ - 2)
	{
		uxnext = ux_[IX(x, y, nw_)];
	}
	else
	{
		uxnext = ux_[IX(x + 1, y, nw_)];
	}
	if (y > nh_ - 2)
	{
		uynext = uy_[IX(x, y, nw_)];
	}
	else
	{
		uynext = uy_[IX(x, y + 1, nw_)];
	}




	diverg = -0.5f * (uxnext - uxpre + uynext - uypre);
	


	double pressurex_pre;
	double pressurey_pre;
	double pressurex_next;
	double pressurey_next;
	if (x < 1) 
	{
		pressurex_pre = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurex_pre = pressure[IX(x-1, y, nw_)];
	}

	if (y < 1)
	{
		pressurey_pre = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurey_pre = pressure[IX(x , y-1, nw_)];
	}

	if (x > nh_ - 2)
	{
		pressurex_next = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurex_next = pressure[IX(x +1, y, nw_)];
	}
	if (y > nh_ - 2)
	{
		pressurey_next = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurey_next = pressure[IX(x, y+1, nw_)];
	}
	//double pressurex_pre = pressure[IX(x - 1, y, nw_)];
	//double pressurey_pre = pressure[IX(x, y - 1, nw_)];
	//double pressurex_next = pressure[IX(x + 1, y, nw_)];
	//double pressurey_next = pressure[IX(x, y + 1, nw_)];
	pressure[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
	
	//for (size_t i = 0; i < 100; i++)
	//{
	//	if ((x + y) % 2 == 0)
	//	{
	//		pressure[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
	//	}
	//	__syncthreads();
	//	if ((x + y) % 2 != 0)
	//	{
	//		pressure[IX(x, y, nw_)] = (diverg + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
	//	}
	//}
}

__global__ void Projection_gpu(double* ux_, double* uy_, double* pressure, int nw_, int nh_) 
{
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//if (tid >= nw_ * nh_) return;

	//int x = tid % nw_;
	//int y = tid / nw_;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	/*if (x < 1 || x > nw_ - 2)
	{
		Set_boundary(ux_, x, y, nw_, nh_);
		Set_boundary(uy_, x, y, nw_, nh_);
		return;
	}
	if (y < 1 || y > nh_ - 2)
	{
		Set_boundary(ux_, x, y, nw_, nh_);
		Set_boundary(uy_, x, y, nw_, nh_);
		return;
	}*/
	
	/*double pressurex_pre = pressure[IX(x - 1, y, nw_)];
	double pressurey_pre = pressure[IX(x, y - 1, nw_)];
	double pressurex_next = pressure[IX(x + 1, y, nw_)];
	double pressurey_next = pressure[IX(x, y + 1, nw_)];*/
	double pressurex_pre;
	double pressurey_pre;
	double pressurex_next;
	double pressurey_next;
	if (x < 1)
	{
		pressurex_pre = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurex_pre = pressure[IX(x - 1, y, nw_)];
	}

	if (y < 1)
	{
		pressurey_pre = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurey_pre = pressure[IX(x, y - 1, nw_)];
	}

	if (x > nh_ - 2)
	{
		pressurex_next = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurex_next = pressure[IX(x + 1, y, nw_)];
	}
	if (y > nh_ - 2)
	{
		pressurey_next = pressure[IX(x, y, nw_)];
	}
	else
	{
		pressurey_next = pressure[IX(x, y + 1, nw_)];
	}
	ux_[IX(x, y,nw_)] -= 0.5f * (pressurex_next - pressurex_pre);
	uy_[IX(x, y,nw_)] -= 0.5f * (pressurey_next - pressurey_pre);

}



__global__ void Advect_gpu_u(double* ux_, double* uy_, double* ux_next, double* uy_next, double* ux_half,double* uy_half ,double timestep, int nw_, int nh_)
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nw_ * nh_) return;

	int x = tid % nw_;
	int y = tid / nw_;*/

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y>=nh_) return;
	if (x < 1 || x > nw_ - 2)
	{
		Set_boundary(ux_next, x, y, nw_, nh_);
		Set_boundary(uy_next, x, y, nw_, nh_);
		return;
	}
	if (y < 1 || y > nh_ - 2)
	{
		Set_boundary(ux_next, x, y, nw_, nh_);
		Set_boundary(uy_next, x, y, nw_, nh_);
		return;
	}


	double xPosPrev = x - 0.5f*timestep * ux_[IX(x, y, nw_)];
	double yPosPrev = y - 0.5f*timestep * uy_[IX(x, y, nw_)];
	xPosPrev = Clamp(xPosPrev, 0.0f, double(nw_));
	yPosPrev = Clamp(yPosPrev, 0.0f, double(nh_));
	//may be not ux_??
	double ux_advected = Interpolation(xPosPrev, yPosPrev, ux_, nw_);
	double uy_advected = Interpolation(xPosPrev, yPosPrev, uy_, nw_);
	//test interpolatation	 
	//add to ux uy
	ux_next[IX(x, y, nw_)] = ux_advected;
	uy_next[IX(x, y, nw_)] = uy_advected;
	ux_half[IX(x, y, nw_)] = ux_advected;
	uy_half[IX(x, y, nw_)] = uy_advected;
	//ux_half[idx(x, y)] = ux_advected;
	//uy_half[idx(x, y)] = uy_advected;

}



__global__ void Refelect_gpu_u(double* ux_, double* uy_, double* ux_half, double* uy_half,double timestep, int nw_, int nh_)
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nw_ * nh_) return;

	int x = tid % nw_;
	int y = tid / nw_;*/

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	if (x < 1 || x > nw_ - 2||y < 1 || y > nh_ - 2)
	{
		Set_boundary(ux_, x, y, nw_, nh_);
		Set_boundary(uy_, x, y, nw_, nh_);
		return;
	}
	ux_[IX(x, y, nw_)] = 2 * ux_[IX(x, y, nw_)] - ux_half[IX(x, y, nw_)];
	uy_[IX(x, y, nw_)] = 2 * uy_[IX(x, y, nw_)] - uy_half[IX(x, y, nw_)];

}

__global__ void Advect_gpu_density(double* density_, double* density_next, double* ux_, double* uy_, double* ux_next, double* uy_next, double timestep, int nw_, int nh_)
{
	/*int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nw_ * nh_) return;

	int x = tid % nw_;
	int y = tid / nw_;*/

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	if (x < 1 || x > nw_ - 2 || y < 1 || y > nh_ - 2)
	{
		Set_boundary(density_next, x, y, nw_, nh_);
		return;
	}

	double xPosPrev = x - timestep * ux_[IX(x, y, nw_)];
	double yPosPrev = y - timestep * uy_[IX(x, y, nw_)];
	xPosPrev = Clamp(xPosPrev, 0.0f, double(nw_));
	yPosPrev = Clamp(yPosPrev, 0.0f, double(nh_));
	double density_advected = Interpolation(xPosPrev, yPosPrev, density_, nw_);
	density_next[IX(x, y, nw_)] = density_advected;
}

__global__ void FVM_Density_gpu(double* density_,double* density_next, double* ux_, double* uy_, double timestep, int nw_, int nh_) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= nw_ || y >= nh_) return;

	if (x < 1 || x > nw_ - 2 || y < 1 || y > nh_ - 2)
	{
		//Set_boundary(density_next, x, y, nw_, nh_);
		return;
	}
	//double ux_leftup =0.25*ux_[IX(x-1, y+1, nw_)]+ 0.25 * ux_[IX(x - 1, y, nw_)]+ 0.25 * ux_[IX(x, y-1, nw_)]+ 0.25 * ux_[IX(x, y, nw_)];
	//double ux_rightup = 0.25 * ux_[IX(x + 1, y, nw_)] + 0.25 * ux_[IX(x, y-1, nw_)] + 0.25 * ux_[IX(x+1, y - 1, nw_)] + 0.25 * ux_[IX(x, y, nw_)];
	//double ux_leftdown = 0.25 * ux_[IX(x - 1, y, nw_)] + 0.25 * ux_[IX(x, y+1, nw_)] + 0.25 * ux_[IX(x-1, y + 1, nw_)] + 0.25 * ux_[IX(x, y, nw_)];
	//double ux_rightdown = 0.25 * ux_[IX(x + 1, y , nw_)] + 0.25 * ux_[IX(x, y+1, nw_)] + 0.25 * ux_[IX(x+1, y + 1, nw_)] + 0.25 * ux_[IX(x, y, nw_)];


	//double uy_leftup = 0.25 * uy_[IX(x - 1, y - 1, nw_)] + 0.25 * uy_[IX(x - 1, y, nw_)] + 0.25 * uy_[IX(x, y - 1, nw_)] + 0.25 * uy_[IX(x, y, nw_)];
	//double uy_rightup = 0.25 * uy_[IX(x + 1, y, nw_)] + 0.25 * uy_[IX(x, y - 1, nw_)] + 0.25 * uy_[IX(x + 1, y - 1, nw_)] + 0.25 * uy_[IX(x, y, nw_)];
	//double uy_leftdown = 0.25 * uy_[IX(x - 1, y, nw_)] + 0.25 * uy_[IX(x, y + 1, nw_)] + 0.25 * uy_[IX(x - 1, y + 1, nw_)] + 0.25 * uy_[IX(x, y, nw_)];
	//double uy_rightdown = 0.25 * uy_[IX(x + 1, y, nw_)] + 0.25 * uy_[IX(x, y + 1, nw_)] + 0.25 * uy_[IX(x + 1, y + 1, nw_)] + 0.25 * uy_[IX(x, y, nw_)];


	double ux_left = 0.5 * ux_[IX(x - 1, y, nw_)] + 0.5 * ux_[IX(x, y, nw_)];
	double ux_right = 0.5 * ux_[IX(x + 1, y, nw_)] + 0.5 * ux_[IX(x, y, nw_)];
	double uy_down = 0.5 * uy_[IX(x, y-1, nw_)] + 0.5 * uy_[IX(x, y, nw_)];
	double uy_up = 0.5 * uy_[IX(x, y+1, nw_)] + 0.5 * uy_[IX(x, y, nw_)];

	//double left_mom = ux_[IX(x - 1, y, nw_)] * density_[IX(x - 1, y, nw_)];
	//double right_mom = ux_[IX(x + 1, y, nw_)] * density_[IX(x + 1, y, nw_)];
	//double up_mom = uy_[IX(x, y + 1, nw_)] * density_[IX(x, y + 1, nw_)];
	//double down_mom = uy_[IX(x, y-1, nw_)] * density_[IX(x, y - 1, nw_)];
	//double current_mom_x = ux_[IX(x, y, nw_)] * density_[IX(x, y, nw_)];
	//double current_mom_y = uy_[IX(x, y, nw_)] * density_[IX(x, y, nw_)];

	//density_[IX(x, y, nw_)] -= (0.5 * (ux_rightup + ux_rightdown) + (-0.5 * (ux_leftup + ux_leftdown))+ (0.5 * (uy_leftup + uy_rightup)) + (-0.5 * (uy_leftdown + uy_rightdown)))*timestep;
	double density_left = 0.5 * density_[IX(x - 1, y, nw_)] + 0.5 * density_[IX(x, y, nw_)];
	double density_right = 0.5 * density_[IX(x + 1, y, nw_)] + 0.5 * density_[IX(x, y, nw_)];
	double density_down = 0.5 * density_[IX(x, y - 1, nw_)] + 0.5 * density_[IX(x, y, nw_)];
	double density_up = 0.5 * density_[IX(x, y+1, nw_)] + 0.5 * density_[IX(x, y, nw_)];

	//double wavespeed = 0;
	//double dissipation_left =0.5* wavespeed * (density_[IX(x - 1, y, nw_)] - density_[IX(x, y, nw_)]);
	//double dissipation_right =0.5* wavespeed * (density_[IX(x + 1, y, nw_)] - density_[IX(x, y, nw_)]);
	//double dissipation_down =0.5* wavespeed * (density_[IX(x, y-1, nw_)] - density_[IX(x, y, nw_)]);
	//double dissipation_up =0.5* wavespeed * (density_[IX(x, y+1, nw_)] - density_[IX(x, y, nw_)]);

	
	//compute momentum and then average
	//double left_term = 0.5*(left_mom+current_mom_x) + dissipation_left;
	//double right_term = 0.5 * (right_mom + current_mom_x) + dissipation_right;
	//double up_term = 0.5 * (up_mom + current_mom_y) + dissipation_up;
	//double down_term = 0.5 * (down_mom + current_mom_y) + dissipation_down;

	//average speed and density then compute momentum
	double left_term = ux_left * density_left ;
	double right_term = ux_right * density_right;
	double up_term = uy_up * density_up  ;
	double down_term = uy_down * density_down ;
	double origin_denisty = density_[IX(x, y, nw_)];

	density_next[IX(x, y, nw_)] =origin_denisty-(right_term-left_term+up_term-down_term) * timestep;
	/*if (density_next[IX(x, y, nw_)] < -0.1) 
	{
		printf("current x:%d current y:%d\n", x, y);
		printf("right vel:%f left vel:%f\n", ux_right, ux_left);
		printf("up vel:%f down vel:%f\n", uy_up, uy_down);
		printf("right density:%f left density:%f\n", density_right, density_left);
		printf("up density:%f down density:%f\n", density_up, density_down);

	}*/

	//using chain rule
	//density_next[IX(x, y, nw_)] = origin_denisty - ((density_right-density_left)*ux_right+(ux_right-ux_left)*density_right+ (density_up - density_down) * uy_up + (uy_up - uy_down) * density_up) * timestep;
	
	//density_next[IX(x, y, nw_)] = origin_denisty - (0.5 * (right_mom + current_mom_x) + 0.5 * (left_mom + current_mom_x) + 0.5 * (up_mom + current_mom_y) + 0.5 * (down_mom + current_mom_y));
	//density_[IX(x, y, nw_)] -=


}


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





int main(int argc, char* argv[])
{
    GLFWwindow* window;
    unsigned width = 1000;
    unsigned height =1000 ;

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
        /**** Initiate Objects Here ****/
        Shader shader("fluid.vs", "fluid.fs");

        FluidScene fluid {1000, 1000};
        FluidVisualizer renderer {&shader, &fluid};

        { // Initialize here
            //fluid.setCircleAtCenter();
        }

        int curKeyState = GLFW_RELEASE;
        int lastKeyState = GLFW_RELEASE;

		FluidData* GPU_Data = fluid.simulator_GPU->GPU_Data;
		FluidData* CPU_Data = fluid.simulator_GPU->CPU_Data;
		int nw = fluid.simulator_GPU->nw_;
		int nh = fluid.simulator_GPU->nh_;
		//int blks = (nw * nh + NUM_THREADS - 1) / NUM_THREADS;
		dim3 blks(64, 64);
		dim3 grid((nw - 1) / blks.x + 1, (nh - 1) / blks.y + 1);
		//int blks = 1;
		cudaError_t cudaStatus;
		int iterTime = 20;

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
					

					//GPU
					fluid.simulator_GPU->InletJetflow(CPU_Data,1);

					fluid.simulator_GPU->CPU_TO_GPU(CPU_Data, GPU_Data);
                   


					//advect
					Advect_gpu_u <<<blks,grid >>> (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, GPU_Data->ux_half, GPU_Data->uy_half,fluid.simulator_GPU->timeStep, nw, nh);
                    cudaDeviceSynchronize();
                    AdvanceTime << <blks,grid >> > (GPU_Data->ux_, GPU_Data->ux_next,nw,nh);
					cudaDeviceSynchronize();
                    AdvanceTime << <blks,grid >> > (GPU_Data->uy_, GPU_Data->uy_next,nw,nh);
					cudaDeviceSynchronize();

			
					//diffuse u
					for (size_t i = 0; i < iterTime; i++)
					{
						diffuse_gpu <<<blks,grid >>> (GPU_Data->ux_, GPU_Data->ux_next, fluid.simulator_GPU->viscosity, nw, nh);
						cudaDeviceSynchronize();
						diffuse_gpu <<<blks,grid >>> (GPU_Data->uy_, GPU_Data->uy_next, fluid.simulator_GPU->viscosity, nw, nh);			
						cudaDeviceSynchronize();
					}
					AdvanceTime << <blks,grid >> > (GPU_Data->ux_, GPU_Data->ux_next, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks,grid >> > (GPU_Data->uy_, GPU_Data->uy_next, nw, nh);
					cudaDeviceSynchronize();

					//project
					SetValue << <blks,grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					cudaDeviceSynchronize();
					for (size_t i = 0; i < iterTime; i++)
					{
						ComputePressure_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
						cudaDeviceSynchronize();
					}
					Projection_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					cudaDeviceSynchronize();


					//reflect
					Refelect_gpu_u << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_half, GPU_Data->uy_half, fluid.simulator_GPU->timeStep, nw, nh);
					cudaDeviceSynchronize();

					//advect
					Advect_gpu_u << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, GPU_Data->ux_half, GPU_Data->uy_half, fluid.simulator_GPU->timeStep, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks,grid >> > (GPU_Data->ux_, GPU_Data->ux_next, nw, nh);
					cudaDeviceSynchronize();
					AdvanceTime << <blks,grid >> > (GPU_Data->uy_, GPU_Data->uy_next, nw, nh);
					cudaDeviceSynchronize();



					//project
					SetValue << <blks,grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					cudaDeviceSynchronize();
					for (size_t i = 0; i < iterTime; i++)
					{
						ComputePressure_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
						cudaDeviceSynchronize();
					}
					Projection_gpu << <blks,grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					cudaDeviceSynchronize();

					

					SetValue << <blks, grid >> > (GPU_Data->pressure, 0.0f, nw, nh);
					cudaDeviceSynchronize();
					for (size_t i = 0; i < iterTime; i++)
					{
						ComputePressure_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
						cudaDeviceSynchronize();
					}
					Projection_gpu << <blks, grid >> > (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->pressure, nw, nh);
					cudaDeviceSynchronize();

					//advect density

                    Advect_gpu_density <<<blks,grid>>> (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, fluid.simulator_GPU->timeStep, nw, nh);
					cudaDeviceSynchronize();
                    AdvanceTime << <blks,grid >> > (GPU_Data->density_, GPU_Data->density_next,nw,nh);
					cudaDeviceSynchronize();
					//diffuse density
					for (size_t i = 0; i < iterTime; i++)
					{
						diffuse_gpu << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, fluid.simulator_GPU->diffk, nw, nh);
						cudaDeviceSynchronize();
					}
					AdvanceTime << <blks,grid >> > (GPU_Data->density_, GPU_Data->density_next, nw, nh);
					cudaDeviceSynchronize();


					//FVM density
					//for (size_t i = 0; i < iterTime; i++)
					//{
						//FVM_Density_gpu << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, fluid.simulator_GPU->timeStep, nw, nh);
						//cudaDeviceSynchronize();
					//}
					//AdvanceTime << <blks,grid >> > (GPU_Data->density_, GPU_Data->density_next, nw, nh);
					//cudaDeviceSynchronize();


					


					//Check_gpu << <blks, grid >> > (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, fluid.simulator_GPU->timeStep, nw, nh);

					fluid.simulator_GPU->GPU_TO_CPU(CPU_Data, GPU_Data);
					printf("doing\n");

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