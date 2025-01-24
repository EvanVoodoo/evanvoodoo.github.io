---
layout: post
title: Creating a GPU-based Particle System
subtitle: Using compute shaders to simulate particles
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [gpu, particles]
author: Evan Lohento
---

# Introduction

Hi, I'm Evan. At the time of writing this, I am a second-year programming student at BUas in the CMGT program. Over the last 8 weeks, I have been working on implementing a particle system, a project I was able to choose myself, with the guidance of lecturers. In this blog post I will explain how I implemented said particle system using compute shaders to simulate the particles, in OpenGL. This article will require some understanding of OpenGL and C++, but nothing too complex.

![BUas Logo](../assets/img/BUas_Logo.png)

## Why on the GPU?

My implementation of this particle system is done on the GPU using compute shaders to simulate particles. This is because the GPU allows for a much larger quantity of particles to be simulated as it can perform computations much faster than the CPU. I specifically chose to use compute shaders in GLSL as I hadn't used them before and was interested in learning more about how to use them.

# Generating Particles

## Particle Emitters

To generate particles, we need particle emitters. These will contain all the info we will need to generate a particle. In this implementation, particle emitter are a component linked to an entity in an ECS. This entity also has a transform, which is what I use to get the position and rotation of my emitter. An ECS isn't necessary for the particle system to work, but I am mentioning it as code snippets may include it. Below is the implementation for the particle emitters.

```cpp
class ParticleEmitter
{
    ParticleEmitter(EmitterType type = EmitterType::SPRITE);
    ParticleEmitter(ParticleEmitter& emitter, EmitterType type = EmitterType::SPRITE);
    ParticleEmitter(glm::vec4 minColor,
                glm::vec4 maxColor,
                glm::vec3 minPosition,
                glm::vec3 maxPosition,
                glm::vec3 minVelocity,
                glm::vec3 maxVelocity,
                glm::vec3 minAccel,
                glm::vec3 maxAccel,
                float minLife,
                float maxLife,
                float spawnInterval,
                uint32_t maxParticles,
                EmitterType type = EmitterType::SPRITE);

    EmitterType emitterType = EmitterType::SPRITE;
    glm::vec4 minColor = glm::vec4(0, 0, 0, 1), maxColor = glm::vec4(1);
    glm::vec3 minPosition = glm::vec3(0), maxPosition = glm::vec3(0);
    glm::vec3 minVelocity = glm::vec3(-1), maxVelocity = glm::vec3(1);
    glm::vec3 minAccel = glm::vec3(0), maxAccel = glm::vec3(0, 0, -0.5f);
    float minLife = 1, maxLife = 10;
    float spawnInterval = 0.00001f, timer = 0;
    uint32_t maxParticles = DEFAULT_MAX_PARTICLES;
    uint32_t drawnParticleCount = 0;

    Buffer particles;
    Buffer freelist;
    Buffer indirectDrawBuffer;
    Buffer drawlist;
};
```

The particle emitter is implemented as a class that contains minimum and maximum values for various attributes, that are then used as a range to select from when generating particles. There are also a couple variables meant just for the emitter. spawnInterval is the time that has to elasp before a particle is spawned, and timer keeps track of the time that has passed. maxParticles is meant to be used to cap the amount of particles that can be alive at once, created by that emitter. 

The buffers at the bottom of the struct are the buffers we will bind to use in the compute shaders. They are just meant to hold the buffer's id but I decided to implement a couple functions to make binding and setting data a bit easier.

```cpp
struct Buffer
{
    uint32_t bufferID;

    Buffer() = default;
    Buffer(uint32_t target, size_t size = 0, const void* data = nullptr, uint32_t usage = 0x88E8);  // value of GL_DYNAMIC_DRAW
    ~Buffer();
    void Bind(uint32_t type) const;
    void BindSSBO(uint32_t index);
    void SetData(size_t size, const void* data) const;
};
```

Later on when we pass this data to the GPU, we'll use a different struct that I called EmitterSettings, so that we only send useful information to the GPU. Below is the implementation for EmitterSettings.

```cpp
struct EmitterSettings
{
    glm::vec4 minColor;
    glm::vec4 maxColor;
    glm::vec3 minOffset;
    float padding1;
    glm::vec3 maxOffset;
    float padding2;
    glm::vec3 minVelocity;
    float padding3;
    glm::vec3 maxVelocity;
    float padding4;
    glm::vec3 minAccel;
    float padding5;
    glm::vec3 maxAccel;
    float padding6;
    glm::mat4 modelMatrix;
    glm::mat4 rotationMatrix;
    float minLife, maxLife;
};
```

## Why is there padding?

You might be confused as to why there is padding in between some of the variables. That is because of the way that GLSL stores its data. Below is the implementation of EmitterSettings in the compute shader.

```glsl
layout (std140, binding = 50) uniform EmitterSettings {
	vec4 minColor, maxColor;
	vec3 minOffset, maxOffset;
	vec3 minVelocity, maxVelocity;
	vec3 minAccel, maxAccel;
	mat4 modelMatrix;
	mat4 rotationMatrix;
	float minLife, maxLife;
} u_emitter;
```

As you can see, there is no padding in the GLSL version of the struct. At least not visibly. Using std140 (or std430) as the layout means that vec3s are treated as vec4s which means they take up 16 bytes (4 float values, each float is 4 bytes). This is because using std140 means that vec4s and vec3s will be aligned to 16 bytes. mat4s are a group of 4 vec4s, so no padding is applied after them. And since there are no more vec4s or vec3s after two float variables, no padding is needed there either. The same thing will apply to the Particle struct later.

## SSBOs



# Conclusion

