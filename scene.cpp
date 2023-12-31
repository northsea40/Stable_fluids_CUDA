#include "scene.hpp"

#include <glm/glm.hpp>

FluidScene::
FluidScene(
    int nw,
    int nh,
    float dx
) :
nw(nw), nh(nh), dx(dx),
width(nw * dx), height(nh * dx)
{
    // Consider to initialize your fluid here
    //initialize simulator
    //simulator = new FluidSimulator(nw,nh,dx);

    simulator_GPU = new FluidSimulator_GPU(nw, nh, dx);
}

void FluidScene::
setCircleAt(glm::vec2 center, float radius) {
    for (int ih = 0; ih < nh; ih++) {
        for (int iw = 0; iw < nw; iw++) {
            glm::vec2 pos = {iw, ih};
            if (glm::distance(pos, center) < radius) {
                simulator->density_[ih * nw + iw] = 1.f;
                simulator->density_next[ih * nw + iw] = 1.f;
            }
        }
    }
}

void FluidScene::
applyImpulsiveVelocity(glm::vec2 pos, glm::vec2 delta_vel) {
    float x = pos.x;
    float y = pos.y;
    int start_x = static_cast<int>(x / dx);
    int start_y = static_cast<int>(y / dx);

    if ((start_x < 0) || (start_x > nw - 1)) { return; }
    if ((start_y < 0) || (start_y > nh - 1)) { return; }

    int iw = static_cast<int>(start_x);
    int ih = static_cast<int>(start_y);

    simulator->ApplyDeltaVelocity(iw, ih, delta_vel);

    // Add delta_vel.x and delta_vel.y to your velocity
}