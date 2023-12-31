#pragma once

#include <cmath>

#include <glm/glm.hpp>

#include "shader.hpp"
#include "scene.hpp"

class FluidVisualizer {
private:
    struct VertexData {
        glm::vec2 position;
        glm::vec2 texCoord;
    };

    struct GLObject {
        GLuint VAO;
        GLuint VBO; std::vector<VertexData> vertices;
        GLuint EBO; std::vector<GLuint> indices;

        // Texture to render the fluid
        GLuint texture; std::vector<float> texData;
        GLuint sampler; const int samplerId = 0;

        GLObject() {
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);
            glGenTextures(1, &texture);
            glGenSamplers(1, &sampler);
        };
        ~GLObject() {
            glDeleteSamplers(1, &sampler);
            glDeleteTextures(1, &texture);
            glDeleteBuffers(1, &EBO);
            glDeleteBuffers(1, &VBO);
            glDeleteVertexArrays(1, &VAO);
        };
        void initTexture() {
            glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        }
        void initData() {
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

            glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                sizeof(VertexData),
                (void*)offsetof(VertexData, position)
            );

            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                sizeof(VertexData),
                (void*)offsetof(VertexData, texCoord)
            );

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        };
    };

    Shader* shader;
    FluidScene* fluid;

    GLObject glo;

public:
    FluidVisualizer(
        Shader* shader,
        FluidScene* fluid
    );
    ~FluidVisualizer() = default;

    void draw();

private:
    void initVertices();
    void initIndices();
    void updateTexture();

    glm::vec4 densityToColor(float density) {
        return {density, density, density, 1.0f};
    }

    glm::vec4 velocityToColor(glm::vec2&& vel) {
        if (std::isnan(vel[0]) || std::isinf(vel[0]) || std::isnan(vel[1]) || std::isinf(vel[1])) {
          
            return {1.0f, 1.0f, 1.0f, 1.0f};
        } else {
            return {std::abs(vel[0]), std::abs(vel[1]), 0.0f, 1.0f};
        }
    };
};