import glm, random
#PyGLM installieren

def generatePerlinVoxelPositions(width, height, depth):
    blockSize = 1
    noiseScale = 20
    amplitude = 20
    offset = random.randrange(0, 1000000)
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                noise = glm.perlin(glm.vec3(x/noiseScale + offset, y/noiseScale + offset, z/noiseScale + offset)) * amplitude
                if noise >= 0.5:
                    data.append([x*blockSize, y*blockSize, z*blockSize])
    return data
