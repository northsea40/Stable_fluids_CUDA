#version 430 core

in VS_OUT {
    vec2 TexCoord;
} fs_in;

uniform sampler2D Tex;
out vec4 OutColor;

void main()
{
    OutColor = texture(Tex, fs_in.TexCoord);
}
