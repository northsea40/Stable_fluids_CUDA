#include"simulator_GPU.h"
#include<algorithm>
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
	cudaMalloc((void**)&(sq->pressure_next), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->ux_half), nw_ * nh_ * sizeof(double));
	cudaMalloc((void**)&(sq->uy_half), nw_ * nh_ * sizeof(double));
}

void FluidSimulator_GPU::CPU_Initialization_ib_helpler(double** uib, double* u) 
{

}

void FluidSimulator_GPU::CPU_Initialization(FluidData* sq)
{
	sq->density_ = new double[nw_ * nh_] {0};
	sq->density_next = new double[nw_ * nh_] {0};
	sq->ux_ = new double[nw_ * nh_] {0};
	sq->ux_next = new double[nw_ * nh_] {0};
	sq->uy_ = new double[nw_ * nh_] {0};
	sq->uy_next = new double[nw_ * nh_] {0};
	sq->pressure= new double[nw_ * nh_] {0};
	sq->ux_half = new double[nw_ * nh_] {0};
	sq->uy_half = new double[nw_ * nh_] {0};

	sq->ux_inner = new double* [nw_ - 2];
	sq->uy_inner = new double* [nw_ - 2];
	sq->ux_next_inner = new double* [nw_ - 2];
	sq->uy_next_inner = new double* [nw_ - 2];

	sq->ux_border = new double* [4];
	sq->uy_border = new double* [4];
	sq->ux_next_border = new double* [4];
	sq->uy_next_border = new double* [4];

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


}
