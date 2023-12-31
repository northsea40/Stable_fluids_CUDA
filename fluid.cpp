#include "fluid.hpp"

FluidVisualizer::
FluidVisualizer(
    Shader* shader,
    FluidScene* fluid
) {
    this->shader = shader;
    this->fluid = fluid;

    this->initVertices();
    this->initIndices();

    this->glo.initTexture();
    this->glo.initData();
}

void FluidVisualizer::
draw() {
    // Update Fluid Data
    updateTexture();

    glBindBuffer(GL_ARRAY_BUFFER, glo.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * this->glo.vertices.size(), this->glo.vertices.data(), GL_STREAM_DRAW);

    GLint previous;
    glGetIntegerv(GL_POLYGON_MODE, &previous);

    this->shader->use();

    // Activate the texture sampler with the texture to render the fluid
    this->shader->setInt("tex", this->glo.samplerId);
    glActiveTexture(GL_TEXTURE0 + this->glo.samplerId);
    glBindTexture(GL_TEXTURE_2D, this->glo.texture);
    glBindSampler(this->glo.samplerId, this->glo.sampler);

    glBindVertexArray(glo.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, glo.VBO);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FLAT); // We want flat mode
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(glo.indices.size()), GL_UNSIGNED_INT, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glPolygonMode(GL_FRONT_AND_BACK, previous); // restore previous mode
}

void FluidVisualizer::
initVertices() {
    // A simple square which is applied to scree space directly
    this->glo.vertices = {
        { glm::vec2( 1.0f,  1.0f), glm::vec2(1.0f, 1.0f) },
        { glm::vec2( 1.0f, -1.0f), glm::vec2(1.0f, 0.0f) },
        { glm::vec2(-1.0f,  1.0f), glm::vec2(0.0f, 1.0f) },
        { glm::vec2(-1.0f, -1.0f), glm::vec2(0.0f, 0.0f) }
    };
}

void FluidVisualizer::
initIndices() {
    // A simple square which is applied to scree space directly
    this->glo.indices = {
        0, 1, 2,
        3, 1, 2
    };
}

void FluidVisualizer::
updateTexture() {
    const int nw = fluid->nw;
    const int nh = fluid->nh;

    this->glo.texData.resize(4 * nh * nw);

    for(int iw = 0; iw < nw; ++iw) {
        for(int ih = 0; ih < nh; ++ih) {
            const int pos = 4 * (ih * nw + iw);

           // glm::vec4 color = velocityToColor(fluid->getVelocity(iw, ih));
            glm::vec4 color = densityToColor(fluid->getDensity(iw, ih));

            this->glo.texData[pos + 0] = color[0];
            this->glo.texData[pos + 1] = color[1];
            this->glo.texData[pos + 2] = color[2];
            this->glo.texData[pos + 3] = color[3];
        }
    }

    glBindTexture(GL_TEXTURE_2D, this->glo.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, nw, nh, 0, GL_RGBA, GL_FLOAT, this->glo.texData.data());
}