#include"simulator.h"
#include<algorithm>

FluidSimulator::FluidSimulator(int nw, int nh, float dx) :m_num(nw* nh)
{
	this->nw_ = nw;
	this->nh_ = nh;
	ux_ = new float[nw * nh];
	uy_ = new float[nw * nh];
	ux_advect = new float[nw * nh];
	uy_advect = new float[nw * nh];
	ux_half = new float[nw * nh];
	uy_half = new float[nw * nh];
	diverg = new float[nw * nh];
	pressure = new float[nw * nh];
	density_ = new float[nw * nh];
	density_next = new float[nw * nh];
	timeStep = 1.0f;
	for (auto y = 0; y < nh_; y++) {
		for (auto x = 0; x < nw_; x++) {
			ux_[idx(x, y)] = 0;
			uy_[idx(x, y)] = 0;
			ux_advect[idx(x, y)] = 0;
			uy_advect[idx(x, y)] = 0;
			ux_half[idx(x, y)] = 0;
			uy_half[idx(x, y)] = 0;
			diverg[idx(x, y)] = 0;
			pressure[idx(x, y)] = 0;
			density_[idx(x, y)] = 0.f;
			density_next[idx(x, y)] = 0.f;
		}
	}

	laplaceMat.resize(m_num, m_num);
	partialxMat.resize(m_num, m_num);
	partialyMat.resize(m_num, m_num);
	diffMat.resize(m_num, m_num);
	laplaceMat.setZero();
	partialxMat.setZero();
	partialyMat.setZero();
	diffMat.setZero();

	for (auto y = 0; y < nh_; y++)
	{
		for (auto x = 0; x < nw_; x++)
		{
			int p = idx(x, y);
			laplaceMat.coeffRef(p, p) = 4;
			if (x - 1 >= 0)
			{
				int curidx = idx(x - 1, y);
				laplaceMat.coeffRef(p, curidx) = -1;
			}
			if (x + 1 <= nw_ - 1)
			{
				int curidx = idx(x + 1, y);
				laplaceMat.coeffRef(p, curidx) = -1;
			}
			if (y - 1 >= 0)
			{
				int curidx = idx(x, y - 1);
				laplaceMat.coeffRef(p, curidx) = -1;
			}
			if (y + 1 <= nh_ - 1)
			{
				int curidx = idx(x, y + 1);
				laplaceMat.coeffRef(p, curidx) = -1;
			}
		}
	}
	diffMat = laplaceMat * viscosity * timeStep;

	for (auto y = 0; y < nh_; y++)
	{
		for (auto x = 0; x < nw_; x++)
		{
			int p = idx(x, y);
			if (x - 1 >= 0)
			{
				int curidx = idx(x - 1, y);
				partialxMat.coeffRef(p, curidx) = -0.5f;
			}
			if (x + 1 <= nw_ - 1)
			{
				int curidx = idx(x + 1, y);
				partialxMat.coeffRef(p, curidx) = 0.5f;
			}
		}
	}

	for (auto y = 0; y < nh_; y++)
	{
		for (auto x = 0; x < nw_; x++)
		{
			int p = idx(x, y);
			if (y - 1 >= 0)
			{
				int curidx = idx(x, y-1);
				partialyMat.coeffRef(p, curidx) = -0.5f;
			}
			if (y + 1 <= nh_ - 1)
			{
				int curidx = idx(x, y+1);
				partialyMat.coeffRef(p, curidx) = 0.5f;
			}
		}
	}
}
void FluidSimulator::InletJetflow(float t)
{

	for (auto x = 0; x < 3; x++)
	{
		for (auto y = nh_/2-2; y < nh_ / 2 + 2; y++)
		{
			ux_[idx(x, y)] += timeStep * 0.5;
			density_[idx(x, y)] = 1;
		}
	}
}


void FluidSimulator::advect()
{
	for (auto y = 1; y < nh_-1; y++)
	{
		for (auto x = 1; x < nw_-1; x++)
		{
			//float xPosPrev = x  - 0.5f*timeStep * ux_[idx(x, y)];
			//float yPosPrev = y  - 0.5f*timeStep * uy_[idx(x, y)];
			float xPosPrev = x - timeStep * ux_[idx(x, y)];
			float yPosPrev = y - timeStep * uy_[idx(x, y)];
			xPosPrev=std::clamp(xPosPrev, 0.0f, float(nw_));
			yPosPrev=std::clamp(yPosPrev, 0.0f, float(nh_));
			//may be not ux_??
			float ux_advected = this->Interpolation(xPosPrev, yPosPrev, ux_);
			float uy_advected = this->Interpolation(xPosPrev, yPosPrev, uy_);
			//test interpolatation	 
			//add to ux uy
			ux_advect[idx(x, y)] = ux_advected;
			uy_advect[idx(x, y)] = uy_advected;
			ux_half[idx(x, y)] = ux_advected;
			uy_half[idx(x, y)] = uy_advected;
		}
	}
	set_boundary(ux_advect);
	set_boundary(uy_advect);
	AdvanceTime(ux_, ux_advect);
	AdvanceTime(uy_, uy_advect);



}

void FluidSimulator::reflect() 
{
	for (auto y = 1; y < nh_ - 1; y++)
	{
		for (auto x = 1; x < nw_ - 1; x++)
		{

			ux_advect[idx(x, y)] = 2*ux_[idx(x, y)]-ux_half[idx(x, y)];
			uy_advect[idx(x, y)] = 2 * uy_[idx(x, y)] - uy_half[idx(x, y)];
		}
	}
	set_boundary(ux_advect);
	set_boundary(uy_advect);
	AdvanceTime(ux_, ux_advect);
	AdvanceTime(uy_, uy_advect);

}


void FluidSimulator::advect_density()
{
	for (auto y = 1; y < nh_-1; y++)
	{
		for (auto x = 1; x < nw_-1; x++)
		{
			float xPosPrev = x  - timeStep * ux_[idx(x, y)];
			float yPosPrev = y  - timeStep * uy_[idx(x, y)];
			xPosPrev = std::clamp(xPosPrev, 0.0f, float(nw_));
			yPosPrev = std::clamp(yPosPrev, 0.0f, float(nh_));
			float density_advected = this->Interpolation(xPosPrev, yPosPrev, density_);
			density_next[idx(x, y)] = density_advected;
		}
	}
	set_boundary(density_next);
	AdvanceTime(density_, density_next);
}


void FluidSimulator::diffusion()
{
	auto ux_diff_vector = this->ComputeDiffusionE(ux_advect);
	auto uy_diff_vector = this->ComputeDiffusionE(uy_advect);
	for (size_t i = 0; i < m_num; i++)
	{
		ux_advect[i] = ux_diff_vector(i);
		uy_advect[i] = uy_diff_vector(i);
	}
}


void FluidSimulator::diffusion_paper()
{
	for (auto c = 0; c < 100; c++)
	{
		for (auto y = 1; y < nh_ - 1; y++)
		{
			for (auto x = 1; x < nw_ - 1; x++)
			{
				ux_advect[idx(x, y)] = (ux_[idx(x, y)] + viscosity*(ux_advect[idx(x - 1, y)] + ux_advect[idx(x + 1, y)] + ux_advect[idx(x, y - 1)] + ux_advect[idx(x , y + 1)])) / (1.0f + 4.0f * viscosity);
				uy_advect[idx(x, y)] = (uy_[idx(x, y)] + viscosity*(uy_advect[idx(x - 1, y)] + uy_advect[idx(x + 1, y)] + uy_advect[idx(x, y - 1)] + uy_advect[idx(x , y + 1)])) / (1.0f + 4.0f * viscosity);
			}
		}
		set_boundary(ux_advect);
		set_boundary(uy_advect);
	}

	AdvanceTime(ux_, ux_advect);
	AdvanceTime(uy_, uy_advect);
}

void FluidSimulator::diffusion_density_paper()
{
	for (auto c = 0; c < 40; c++) 
	{
		for (auto y = 1; y < nh_-1; y++)
		{
			for (auto x = 1; x < nw_-1; x++)
			{
				density_next[idx(x, y)] = (density_[idx(x, y)] + diffk * (density_next[idx(x - 1, y)] + density_next[idx(x + 1, y)] + density_next[idx(x, y - 1)] + density_next[idx(x, y + 1)])) / (1.0f+4.0f* diffk);
			}
		}
		set_boundary(density_next);
	}
	AdvanceTime(density_, density_next);
}


void FluidSimulator::diffusion_density()
{
	auto density_diff_vector = this->ComputeDiffusionE(density_);
	for (size_t i = 0; i < m_num; i++)
	{
		density_[i] = density_diff_vector(i);
	}
}



void FluidSimulator::projection()
{
	//compute the divergence
	 auto ux_vector=CopyArrayToVec(ux_advect,m_num);
	 auto uy_vector = CopyArrayToVec(uy_advect, m_num);
	 auto u_partialx = this->partialxMat * ux_vector;
	 auto u_partialy = this->partialyMat * uy_vector;
	 auto u_divergence = u_partialx + u_partialy;
	 auto A = this->laplaceMat;
	 auto b = u_divergence;
	 Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
	 cg.compute(A);
	 auto q = cg.solve(b);
	 Eigen::VectorXf partialq_x = partialxMat * q;
	 Eigen::VectorXf partialq_y = partialyMat * q;
	 for (size_t i = 0; i < m_num; i++)
	 {
		 ux_[i] = ux_advect[i] - partialq_x(i);
		 uy_[i] = uy_advect[i] - partialq_y(i);
	 }
}

void FluidSimulator::projection_paper() 
{
	for (auto y = 1; y < nh_-1; y++)
	{
		for (auto x = 1; x < nw_-1; x++)
		{		
			float ux_pre = ux_advect[idx(x-1, y)];
			float uy_pre = uy_advect[idx(x, y-1)];
			float ux_next = ux_advect[idx(x+1, y)];
			float uy_next = uy_advect[idx(x, y+1)];
			diverg[idx(x, y)] = -0.5f * (ux_next - ux_pre + uy_next - uy_pre);
			pressure[idx(x, y)] = 0;
		}
	}
	set_boundary(diverg);
	set_boundary(pressure);

	for (auto c = 0; c < 30; c++)
	{
		for (auto y = 1; y < nh_-1; y++)
		{
			for (auto x = 1; x < nw_-1; x++)
			{
				float pressurex_pre = pressure[idx(x - 1, y)];
				float pressurey_pre = pressure[idx(x, y - 1)];		
				float pressurex_next = pressure[idx(x + 1, y)];			
				float pressurey_next = pressure[idx(x, y + 1)];
				pressure[idx(x, y)] = (diverg[idx(x, y)] + pressurex_pre + pressurex_next + pressurey_next + pressurey_pre) / 4.0f;
			}
		}
		set_boundary(pressure);
	}

	for (auto y = 1; y < nh_-1; y++)
	{
		for (auto x = 1; x < nw_-1; x++)
		{

			float pressurex_pre = pressure[idx(x - 1, y)];
			float pressurey_pre = pressure[idx(x, y - 1)];
			float pressurex_next = pressure[idx(x + 1, y)];
			float pressurey_next = pressure[idx(x, y + 1)];
			ux_[idx(x, y)] -=   0.5f * (pressurex_next - pressurex_pre);
			uy_[idx(x, y)] -=   0.5f *(pressurey_next - pressurey_pre);
		}
	}
	set_boundary(ux_);
	set_boundary(uy_);
}


void FluidSimulator::set_boundary(float* u)
{
	for (auto y = 1; y < nh_-1; y++)
	{
		u[idx(0, y)] = 0.5f* u[idx(1, y)];
		u[idx(nw_-1, y)]= 0.5f * u[idx(nw_ - 2, y)];
	}
	for (auto x = 1; x < nw_-1; x++)
	{
		u[idx(x, 0)] = 0.5f * u[idx(x, 1)];
		u[idx(x, nh_-1)] = 0.5f * u[idx(x,nh_-2)];
	}
	u[idx(0, 0)] = 0.25f * u[idx(1, 0)] + 0.25f * u[idx(0, 1)];
	u[idx(nw_-1, 0)] = 0.25f * u[idx(nw_-2, 0)] + 0.25f * u[idx(nw_-1, 1)];
	u[idx(0, nh_-1)] = 0.25f * u[idx(0, nh_-2)] + 0.25f * u[idx(1, nh_-1)];
	u[idx(nw_-1, nh_-1)] = 0.25f * u[idx(nw_-2, nh_-1)] + 0.25f * u[idx(nw_-1, nh_-2)];	

}
void FluidSimulator::AdvanceTime(float* oldarray, float* newarray) 
{
	for (auto y = 0; y < nh_; y++)
	{
		for (auto x = 0; x < nw_; x++)
		{
			oldarray[idx(x, y)] = newarray[idx(x, y)];
		}
	}

}

Eigen::VectorXf FluidSimulator::ComputeDiffusionE(float* u) 
{
	Eigen::VectorXf x(m_num);
	Eigen::SparseMatrix<float> A(m_num, m_num);
	A = this->diffMat;
	auto b = CopyArrayToVec(u, m_num);
	//create identity
	Eigen::SparseMatrix<float>I(m_num, m_num);
	I.setIdentity();
	//set laplace matrix
	A = I - A;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>,Eigen::Lower | Eigen::Upper> cg;
	cg.compute(A);
	x = cg.solve(b);
	return x;
}


float FluidSimulator::Interpolation(float x, float y,float* u) 
{
	// if the dx is not equal to 1 need to change
	int x_ceil = std::ceil(x);
	int x_floor = std::floor(x);
	int y_ceil = std::ceil(y);
	int y_floor = std::floor(y);
	int left_down_idx = idx(x_floor, y_floor);
	int left_up_idx= idx(x_floor, y_ceil);
	int right_down_idx = idx(x_ceil, y_floor);
	int right_up_idx = idx(x_ceil, y_ceil);
	float t1 = y - float(y_floor);
	float t0 = 1 - t1;
	float s1 = x - float(x_floor);
	float s0 = 1 - s1;
	return s0 * (t0 * u[left_down_idx] + t1 * u[left_up_idx]) + s1 * (t0 * u[right_down_idx] + t1 * u[right_up_idx]);
	/*float tx = float(x_ceil) - x;
    float u_down = (1-tx) * u[left_down_idx] + ( tx) * u[right_down_idx];
	float u_up = (1-tx) * u[left_up_idx] + (tx) * u[right_up_idx];
	float ty = float(y_ceil) - y;
	return (ty) * u_down + (1-ty) * u_up;*/
}

Eigen::VectorXf FluidSimulator::CopyArrayToVec(float* arr, int copynum) 
{
	Eigen::VectorXf ret(copynum);
	for (size_t i = 0; i < copynum; i++)
	{
		ret(i) = arr[i];
	}
	return ret;
}