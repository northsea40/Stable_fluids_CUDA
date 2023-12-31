#pragma once

#include <vector>
#include <glm/vec2.hpp>
#include"simulator.h"
#include"simulator_GPU.h"


class FluidScene {
    friend class FluidSimulator;
    friend class FluidSimulator_GPU;
public:
    const float dx;
    const int nw;
    const int nh;

    const float width;
    const float height;

    FluidSimulator* simulator;
    FluidSimulator_GPU* simulator_GPU;

public:
    FluidScene(int nw, int nh, float dx = 1.0f);
    ~FluidScene() = default;

    // Initialize a circle with density of 1 in the fluid scene
    void setCircleAt(glm::vec2 center, float radius);
    void setCircleAtCenter() {
        setCircleAt({ nw / 2, nh / 2 }, std::min(nw, nh) / 6.0f);
    }

    // Some functions that might be helpful
    constexpr size_t idxFromCoord(int iw, int ih) { return ih * nw + iw; }
    constexpr size_t idxLeftFromCoord(int iw, int ih) { return ih * (nw + 1) + iw; }
    constexpr size_t idxRightFromCoord(int iw, int ih) { return ih * (nw + 1) + iw + 1; }
    constexpr size_t idxDownFromCoord(int iw, int ih) { return ih * nw + iw; }
    constexpr size_t idxUpFromCoord(int iw, int ih) { return (ih + 1) * nw + iw; }

    float getDensity(int iw, int ih) 
    {
        //return simulator->get_density(iw, ih); 
        return simulator_GPU->get_density(iw, ih);
    };
    glm::vec2 getVelocity(int iw, int ih) 
    {
        //return simulator->get_velocity(iw, ih);
        return simulator_GPU->get_velocity(iw, ih);
    }

    void applyImpulsiveVelocity(glm::vec2 pos, glm::vec2 delta_vel);
    void step() {
      /*  simulator->InletJetflow(1);
        simulator->diffusion_paper();
        simulator->projection_paper();
        simulator->advect();
        simulator->projection_paper();
        simulator->reflect();
        simulator->advect();
        simulator->projection_paper();
        simulator->projection_paper();
        simulator->diffusion_density_paper();
        simulator->advect_density();*/
    }

private:
    // Consider your data structure to save fluid properties...
};