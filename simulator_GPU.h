#pragma once
#include <glm/vec2.hpp>
#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <glm/glm.hpp>


typedef struct FluidData
{
	double* density_next;
	double* density_;

	double* ux_;
	double* uy_;

	double* ux_next;
	double* uy_next;

	double* ux_half;
	double* uy_half;

	double* pressure;
	double* pressure_next;
} FluidData;

class FluidSimulator_GPU {
public:
	/*double* ux_;
	double* uy_;
	double* ux_next;
	double* uy_next;
	double* density_;
	double* density_next;*/
	int nw_;
	int nh_;
	const int m_num;
	double timeStep=1.0;
	double dx = 1.0f;
	double viscosity = 0.00001;
	double diffk = 0.00001;
	FluidData* CPU_Data;
	FluidData* GPU_Data;
	//const int NUM_THREADS;


public:
	FluidSimulator_GPU(int nw, int nh, double dx);
	inline unsigned idx(int x, int y) {
		return y * nw_ + x;
	}

	void GPU_Initialization(FluidData* sq);
	void CPU_Initialization(FluidData* sq);
	void GPU_TO_CPU(FluidData* sq_cpu, FluidData* sq_gpu);
	void CPU_TO_GPU(FluidData* sq_cpu, FluidData* sq_gpu);
	void InletJetflow(FluidData* sq_cpu,double t);
	void AdvanceTime(double* oldarray, double* newarray);
	void reflect();
	void Simulator_GPU_Step();
	void FluidSquareStep();
	void ApplyDeltaVelocity(int x, int y, glm::vec2 delta_velocity)
	{
		CPU_Data->ux_[idx(x, y)] += 5.0f * delta_velocity.x;
		CPU_Data->uy_[idx(x, y)] += 5.0f * delta_velocity.y;
		//	std::cout << ux_[idx(x, y)] << " " << uy_[idx(x, y)] << std::endl;
	}

	void setCircleAt(glm::vec2 center, double radius);
	void setCircleAtCenter()
	{
		setCircleAt({ nw_ / 2, nh_ / 2 }, std::min(nw_, nh_) / 6.0f);
	}

	glm::vec2 get_velocity(int x, int y) 
	{
		return glm::vec2(CPU_Data->ux_[idx(x, y)], CPU_Data->uy_[idx(x, y)]);
	}
	double get_density(int x, int y) {
		return CPU_Data->density_[idx(x, y)];
	}

};