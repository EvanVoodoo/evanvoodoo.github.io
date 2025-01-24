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

An SSBO (or Shader Storage Buffer Object) is a type of buffer you can send to the GPU. I use them in my implementation for most of my buffers as they allow for up to 128MB of data, and can be read and written to by the GPU. This is perfect for simulating a ton of particles at once as, in our use case, we need both a lot of space to store the particles, and they need to be modifiable by the GPU.

## Emitting Particles

### Spawning Particles

There are two steps to emitting particles. First is spawning them in. For that we first bind the correct compute shader.

```cpp
m_particleSpawnShader->Activate();
```

After binding the shader program, we loop through all the emitters in the scene, and update their timers. Dividing the timer by the spawn interval gives us the amount of particles that are spawning this frame. If no particles spawn this frame, we can skip this loop.

```cpp
emitter.timer += dt;
emitter.spawnInterval = glm::max(emitter.spawnInterval, std::numeric_limits<float>::epsilon());
uint32_t particlesToSpawn = static_cast<uint32_t>(emitter.timer / emitter.spawnInterval);
emitter.timer = mod(emitter.timer, emitter.spawnInterval);

if (particlesToSpawn <= 0) continue;
```

We then bind the buffers and uniforms necessary for the compute shader to spawn in particles. However, we still need to calculate how many work groups we will need for the glDispatchCompute call. We get the work group size from the compute shader and the particles that need to spawn this frame and use them to calculate the minimum number of groups needed to spawn all the particles.

```cpp
int QueryLinearWorkGroupSize(std::shared_ptr<Shader> shader)
{
    GLint workGroupSize[3];
    glGetProgramiv(shader->GetProgram(), GL_COMPUTE_WORK_GROUP_SIZE, workGroupSize);
    return workGroupSize[0];  // Return the x-dimension of the work group size
}
```
```cpp
emitter.particles.BindSSBO(0);
emitter.freelist.BindSSBO(1);

int workGroupSize = QueryLinearWorkGroupSize(m_particleSpawnShader);
int numWorkGroups =
    ((particlesToSpawn + particlesPerThread * workGroupSize - 1) / (particlesPerThread * workGroupSize));
glDispatchCompute(numWorkGroups, 1, 1);
```

Inside the compute shader, we use a PRNG (pseudo-random number generator) to get a pseudo-random float between 0 and 1 to use to randomly select a value between the given minimum and maximum values for each attribute. 

```glsl
uint rand_xorshift()
{
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float random() {
	return float(wang_hash(rand_xorshift())) * (1.0 / 4294967296.0);
}

void MakeParticle(out Particle particle) {
	particle.life = max(0.1, mix(u_emitter.minLife, u_emitter.maxLife, random()));
	particle.initialLife = particle.life;

	particle.pos = u_emitter.modelMatrix[3].xyz;
	vec4 offset = vec4(
						vec3(
							mix(u_emitter.minOffset.x, u_emitter.maxOffset.x, random()),
							mix(u_emitter.minOffset.y, u_emitter.maxOffset.y, random()), 
							mix(u_emitter.minOffset.z, u_emitter.maxOffset.z, random())
						),
						1
				  );
	particle.pos += vec3(u_emitter.rotationMatrix * offset);
	particle.scale.xy = vec2(0.1);

	particle.velocity.x = mix(u_emitter.minVelocity.x, u_emitter.maxVelocity.x, random());
	particle.velocity.y = mix(u_emitter.minVelocity.y, u_emitter.maxVelocity.y, random());
	particle.velocity.z = mix(u_emitter.minVelocity.z, u_emitter.maxVelocity.z, random());
	particle.velocity = vec3(u_emitter.rotationMatrix * vec4(particle.velocity, 1));
	
	particle.accel.x = mix(u_emitter.minAccel.x, u_emitter.maxAccel.x, random());
	particle.accel.y = mix(u_emitter.minAccel.y, u_emitter.maxAccel.y, random());
	particle.accel.z = mix(u_emitter.minAccel.z, u_emitter.maxAccel.z, random());
	particle.accel = vec3(u_emitter.rotationMatrix * vec4(particle.accel, 1));

	particle.startColor = particle.color = u_emitter.minColor;
	particle.endColor = u_emitter.maxColor;
}

void SpawnParticle(uint index) {
	if (index >= u_particlesToSpawn)
		return;
	// undo decrement and return if nothing in freelist
	int freeListIndex = atomicAdd(freelist.count, -1) - 1;
	if (freeListIndex < 0) {
		atomicAdd(freelist.count, 1);
		return;
	}

	int particleIndex = freelist.indices[freeListIndex];

	rng_state = particleIndex;

	if (u_useShape && shape.verticesCount > 0) {
		MakeParticleShape(particles[particleIndex], shape.vertices[particleIndex % shape.verticesCount].xyz);
	}
	else {
		MakeParticle(particles[particleIndex]);
	}
}

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main() {
	uint index = gl_GlobalInvocationID.x * u_particlesPerThread;

	for (int i = 0; i < u_particlesPerThread; i++) {
		SpawnParticle(index + i);
	}
}
```

### Updating Particles

To update particles, we do the same thing as with spawning them, except we bind two more SSBOs. We also make sure to clear the count in indirectDrawBuffer.

```glsl
void SimulateParticle(inout Particle particle, int index) {
    if (particle.life > 0) {
        particle.pos += particle.velocity * u_dt;
        particle.velocity += particle.accel * u_dt;
        particle.life -= u_dt;

        if (particle.life <= 0.0) {
            particle.color.a = 0.0; // make the particle invisible
            freelist.indices[atomicAdd(freelist.count, 1)] = index;
        }

        else {
            float lifeRatio = particle.life / particle.initialLife;
            particle.color = mix(particle.endColor, particle.startColor, lifeRatio);
            if (particle.life <= particle.initialLife * 0.1) {
                particle.scale = mix(vec2(0), vec2(0.2), particle.life / (particle.initialLife * 0.2));
            }
            atomicAdd(indirectCommand.count, 1);
            if (dot(u_forwardDir, normalize(particle.pos.xyz - u_viewPos)) > u_cullingMinAngle) {
                uint drawIndex = atomicAdd(indirectCommand.instanceCount, 1);
                drawIndices[drawIndex] = index;
            }
        }
    }
}

shared Particle localParticles[128];

void UpdateParticle(uint index) {
    if (index >= particles.length())
        return;
    
    SimulateParticle(particles[index], int(index));
}

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint globalIndex = gl_GlobalInvocationID.x * u_particlesPerThread;

    for (int i = 0; i < u_particlesPerThread; i++) {
        UpdateParticle(globalIndex + i);
    }
}
```

## Rendering Particles

Rendering particles which are sprites is fairly easy since all you need is a quad. We can create a single VAO for the quad and reuse it for every single particle since they are all quads. I added two variables to the ParticleEmitter class: m_VAO and m_VBO.

```cpp
ParticleEmitter::ParticleEmitter()
{
    ...
    float quadVertices[] = {-0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}
```

I bind the VAO created here to render quads for the particles. In the render loop, I loop through all the emitters in the scene, activate my particle shader program, set uniforms in the program, bind the VAO and SSBOs, and call glDrawArraysInstanced.

```cpp
m_particleSpriteShader->Activate();
m_particleSpriteShader->GetParameter("u_viewProj")->SetValue(vp);
m_particleSpriteShader->GetParameter("u_cameraRight")->SetValue(cameraRight);
m_particleSpriteShader->GetParameter("u_cameraUp")->SetValue(cameraUp);
emitter.particles.BindSSBO(0);
emitter.drawlist.BindSSBO(1);
glBindVertexArray(emitter.GetVAO());
m_particleSpriteShader->GetParameter("u_sprite")->SetValue(*emitter.GetParticleTexture()->Image);
glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, emitter.maxParticles);
```

```glsl
#version 460 core

struct Particle {
	vec4 color;
    vec4 startColor;
    vec4 endColor;
	vec3 pos;
	vec3 velocity;
	vec3 accel;
	vec2 scale;
	float life;
	float initialLife;
};

layout(std430, binding = 0) readonly restrict buffer Particles {
    Particle particles[];
};

layout(std430, binding = 1) restrict buffer DrawIndices {
    uint drawIndices[];
};

layout(location = 0) in vec2 aPos; // in [-0.5, 0.5]

uniform mat4 u_viewProj;
uniform vec3 u_cameraRight;
uniform vec3 u_cameraUp;

out vec2 vTexCoord;
out vec4 vColor;

void main() {
    vTexCoord = aPos + 0.5;

    int index = gl_InstanceID;

    Particle particle = particles[index];
    vColor = particle.color;

    vec3 vertexPosition =
        particle.pos.xyz + 
        u_cameraRight * aPos.x * particle.scale.x +
        u_cameraUp * aPos.y * particle.scale.y;

    gl_Position = u_viewProj * vec4(vertexPosition, 1.0);
}
```

```glsl
#version 460 core

in vec2 vTexCoord;
in vec4 vColor;

layout(location = 3, binding = 0) uniform sampler2D u_sprite;

out vec4 fragColor;

void main() {
    vec4 texColor = texture(u_sprite, vTexCoord);
    if (texColor.a <= 0.0 || vColor.a <= 0.0) discard;
    fragColor = texColor * vColor;
}
```

After adding a couple other elements to the scene, these are the particles that get created by our system.

<p align=center>
    <img src="../assets/img/PRNG_WANG_HASH_XOR.gif">
</p>

# Conclusion

In conclusion, this GPU-based particle system works very well and can handle a large amount of particles quite easily. In my own testing, I was easily able to get over a million particles running at around 120 fps.

Now there are a lot of things that could be added to this system. Personally, I have also made it able to render models, and I've added menus with ImGui to allow for more control while the application is running.

<p align=center>
    <img src="../assets/img/MeshParticles.gif">
</p>

<p align=center>
    <img src="../assets/img/CustomizableParticleEmitter.gif">
</p>

There are also a lot of optimizations that I could've added to my code, had I not run out of time. I even left some of it in my code, in hopes of using it in the future when I have more time, and better understand how to use the GPU.

Thanks for reading! Feel free to reach out if you have any questions or want to show me what you’re working on!
