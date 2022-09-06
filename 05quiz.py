import taichi as ti

ti.init(arch=ti.gpu)

n = 512
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))


gui = ti.GUI("quiz", (n, n))

@ti.func
def frac(x):
    return x - ti.floor(x)

@ti.kernel
def paint(t: ti.f32):
    for i_, j_ in pixels:
        c = 0.0
        levels = 7
        for k in range(levels):
            block_size = 2 * 2 ** k
            i = i_ + t
            j = j_ + t
            p = i % block_size / block_size
            q = j % block_size / block_size
            i = i // block_size
            j = j // block_size
            brightness = (0.7 - ti.Vector([p - 0.5, q - 0.5]).norm()) * 2
            weight = 0.5 ** (levels - k - 1) * brightness
            c += frac(ti.sin(float(i * 8 + j * 42 + t * 1e-4)) * 128) * weight

        pixels[i_, j_] = ti.Vector([c, c * 0.8, c])


t = 0.0


while True:
    t += 0.1
    paint(t)

    gui.set_image(pixels)
    gui.show()
