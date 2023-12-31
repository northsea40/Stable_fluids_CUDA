#version 430 core

layout (location = 0) in vec2 Pos;
layout (location = 1) in vec2 TexCoord;

out VS_OUT {
    vec2 TexCoord;
} vs_out;

void main() {
    vs_out.TexCoord = TexCoord;
    gl_Position = vec4(Pos, 0.0, 1.0);
}