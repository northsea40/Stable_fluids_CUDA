#pragma once
#include <glm/vec2.hpp>
#include<iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

class FluidSimulator {
public:
	//some data;
	float* ux_;
	float* uy_;
	float* ux_advect;
	float* uy_advect;
	float* ux_half;
	float* uy_half;
	float* diverg;
	float* pressure;
	float* density_;
	float* density_next;
	int nw_;
	int nh_;
	const int m_num;
	float timeStep;
	float dx=1.0f;
	float viscosity = 0.01f;
	float diffk = 0.001f;
	Eigen::SparseMatrix<float> laplaceMat;
	Eigen::SparseMatrix<float> diffMat;
	Eigen::MatrixXf partialxMat;
	Eigen::MatrixXf partialyMat;
    //TODO: you need to define other fluid variables here


public:
	FluidSimulator(int nw, int nh, float dx);
	inline unsigned idx(int x, int y) {
		return y * nw_ + x;
	}
	void advect();
	void advect_density();
	void diffusion();
	void diffusion_paper();
	void diffusion_density();
	void diffusion_density_paper();
	void projection();
	void projection_paper();
	void set_boundary(float* u);
	void InletJetflow(float t);
	void AdvanceTime(float* oldarray,float* newarray);
	void reflect();
	float Interpolation(float x, float y, float* u);
	Eigen::VectorXf ComputeDiffusionE(float* u);
	Eigen::VectorXf CopyArrayToVec(float* arr, int copynum);
	void ApplyDeltaVelocity(int x, int y, glm::vec2 delta_velocity) {
		
		ux_[idx(x, y)] += 5.0f*delta_velocity.x;
		uy_[idx(x, y)] += 5.0f*delta_velocity.y;
	//	std::cout << ux_[idx(x, y)] << " " << uy_[idx(x, y)] << std::endl;
		

	}

	glm::vec2 get_velocity(int x, int y) {
		return glm::vec2(ux_[idx(x, y)], uy_[idx(x, y)]);
	}
	float get_density(int x, int y) {
		return density_[idx(x, y)];
	}

	
	

};
