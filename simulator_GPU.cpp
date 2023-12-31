#include"simulator_GPU.h"
#include<algorithm>
#include "device_functions.h"
const int NUM_THREADS = 16;



FluidSimulator_GPU::FluidSimulator_GPU(int nw, int nh, double dx):m_num(nw* nh)
{
	this->nh_ = nh;
	this->nw_ = nw;
	this->dx = dx;
	this->CPU_Data = new FluidData();
	this->GPU_Data = new FluidData();
	GPU_Initialization(GPU_Data);
	CPU_Initialization(CPU_Data);
	setCircleAtCenter();
	CPU_TO_GPU(CPU_Data, GPU_Data);
}

void FluidSimulator_GPU::GPU_TO_CPU(FluidData* sq_cpu, FluidData* sq_gpu)
{
	cudaError_t cudaStatus;
	cudaMemcpy(sq_cpu->density_next, sq_gpu->density_next, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);
	cudaMemcpy(sq_cpu->density_,sq_gpu->density_, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);

	cudaMemcpy(sq_cpu->ux_,sq_gpu->ux_, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);
	cudaMemcpy(sq_cpu->uy_, sq_gpu->uy_, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);

	cudaMemcpy(sq_cpu->ux_next, sq_gpu->ux_next, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);
	cudaMemcpy(sq_cpu->uy_next, sq_gpu->uy_next, sizeof(double) * nw_ * nh_, cudaMemcpyDeviceToHost);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "GPU_TO_CPU: %s\n", cudaGetErrorString(cudaStatus));
	}
}

void FluidSimulator_GPU::CPU_TO_GPU(FluidData* sq_cpu, FluidData* sq_gpu) 
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(sq_gpu->density_next, sq_cpu->density_next, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaMemcpy(sq_gpu->density_, sq_cpu->density_, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);

	cudaMemcpy(sq_gpu->ux_, sq_cpu->ux_, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);
	cudaMemcpy(sq_gpu->uy_, sq_cpu->uy_, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);

	cudaMemcpy(sq_gpu->ux_next, sq_cpu->ux_next, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);
	cudaMemcpy(sq_gpu->uy_next, sq_cpu->uy_next, sizeof(double) * nw_ * nh_, cudaMemcpyHostToDevice);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CPU_TO_GPU: %s\n", cudaGetErrorString(cudaStatus));
	}
}


void FluidSimulator_GPU::GPU_Initialization(FluidData* sq)
{
	cudaMalloc((void**)&(sq->density_), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->density_next), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->ux_), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->ux_next), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->uy_), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->uy_next), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->pressure), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->ux_half), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->uy_half), nw_ * nh_ * sizeof(double));
}

void FluidSimulator_GPU::CPU_Initialization(FluidData* sq)
{
	sq->density_ = new double[nw_ * nh_]();
	sq->density_next = new double[nw_ * nh_]();
	sq->ux_ = new double[nw_ * nh_]();
	sq->ux_next = new double[nw_ * nh_]();
	sq->uy_ = new double[nw_ * nh_]();
	sq->uy_next = new double[nw_ * nh_]();
	sq->pressure= new double[nw_ * nh_]();
	sq->ux_half = new double[nw_ * nh_]();
	sq->uy_half = new double[nw_ * nh_]();
}


void FluidSimulator_GPU::setCircleAt(glm::vec2 center, double radius)
{
	for (int ih = 0; ih < nh_; ih++) {
		for (int iw = 0; iw < nw_; iw++)
		{
			glm::vec2 pos = { iw, ih };
			if (glm::distance(pos, center) < radius)
			{
				CPU_Data->density_[ih * nw_ + iw] = 1.0;
				CPU_Data->density_next[ih * nw_ + iw] = 1.0;
			}
		}
	}
}

void  FluidSimulator_GPU::InletJetflow(FluidData* sq_cpu,double t)
{
	for (auto x = 0; x < 6; x++)
	{
		for (auto y = nh_ / 2 - 4; y < nh_ / 2 + 4; y++)
		{
			sq_cpu->ux_[idx(x, y)] += timeStep * 0.5;
			sq_cpu->density_[idx(x, y)] = 1;
		}
	}
}

void FluidSimulator_GPU::FluidSquareStep() 
{
	//TODO

	//int blks = (nw_ * nh_ + NUM_THREADS - 1) / NUM_THREADS;
	//advect_gpu_u <<<blks, NUM_THREADS >>> (GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, timeStep, nw_, nh_);
	//cudaDeviceSynchronize();

	//AdvanceTime(GPU_Data->ux_, GPU_Data->ux_next);
	//AdvanceTime(GPU_Data->uy_, GPU_Data->uy_next);

	//advect_gpu_density <<<blks, NUM_THREADS>>> (GPU_Data->density_, GPU_Data->density_next, GPU_Data->ux_, GPU_Data->uy_, GPU_Data->ux_next, GPU_Data->uy_next, timeStep, nw_, nh_);
	//AdvanceTime(GPU_Data->density_, GPU_Data->density_next);

	//GPU_TO_CPU(GPU_Data, CPU_Data);


	//ALL
	//diffuse_gpu <<<blks, NUM_THREADS>>>  (1, sq->Vx0, sq->Vx, visc, dt, 4, N);
	//cudaDeviceSynchronize();


	//diffuse_gpu <<<blks, NUM_THREADS>>> (2, sq->Vy0, sq->Vy, visc, dt, 4, N);
	//cudaDeviceSynchronize();



	//project_gpu <<<blks, NUM_THREADS>>> (sq->Vx0, sq->Vy0, sq->Vx, sq->Vy, 4, N);
	//cudaDeviceSynchronize();



	//advect_gpu  <<<blks, NUM_THREADS>>> (1, sq->Vx, sq->Vx0, sq->Vx0, sq->Vy0, dt, N);
	//cudaDeviceSynchronize();



	//advect_gpu  <<<blks, NUM_THREADS>>> (2, sq->Vy, sq->Vy0, sq->Vx0, sq->Vy0, dt, N);
	//cudaDeviceSynchronize();



	//project_gpu <<<blks, NUM_THREADS>>> (sq->Vx, sq->Vy, sq->Vx0, sq->Vy0, 4, N);
	//cudaDeviceSynchronize();


	//diffuse_gpu <<<blks, NUM_THREADS>>> (0, sq->density0, sq->density, diff, dt, 4, N);
	//cudaDeviceSynchronize();

	//advect_gpu  <<<blks, NUM_THREADS>>> (0, sq->density, sq->density0, sq->Vx, sq->Vy, dt, N);

}
