import taichi as ti
import handy_shader_functions as hsf

ti.init(arch=ti.gpu)
res_x = 800
res_y = 450

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))


@ti.func
def smin(a, b, k):
    h = hsf.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return hsf.mix(b, a, h) - k * h * (1.0 - h)


@ti.func
def smax(a, b, k):
    return -smin(-a, -b, k)


@ti.func
def rotmat(a):
    return ti.Vector([[ti.cos(a), ti.sin(a)], [-ti.sin(a), ti.cos(a)]])


@ti.func
def shoesDist(p):
    op = p
    d = 1e4
    p[1] -= 1.5

    # right shoe
    op = p
    p -= ti.Vector([-0.5, -0.6, -0.9])
    p_yz = ti.Vector([p[1], p[2]])
    p_yz = rotmat(-0.7) @ p_yz
    p[1] = p_yz[0]
    p[2] = p_yz[1]

    p_xz = ti.Vector([p[0], p[2]])
    p_xz = rotmat(0.1) @ p_xz
    p[0] = p_xz[0]
    p[2] = p_xz[1]
    d = ti.min(d, -smin(p[1], -(p * ti.Vector([1.6, 1.0, 1.0])).norm() - 0.64), 0.2)
    p = op

    # left shoe
    op = p
    p -= ti.Vector([0.55, -0.8, 0.4])
    p[0] = -p[0]
    p_yz = ti.Vector([p[1], p[2]])
    p_yz = rotmat(1.4) @ p_yz
    p[1] = p_yz[0]
    p[2] = p_yz[1]
    d = ti.min(d, -smin(p[1], -(p * ti.Vector([1.6, 1.0, 1.0])).norm() - 0.73), 0.2)
    p = op
    return d


@ti.func
def sceneDist(p):
    op = p
    d = shoesDist(p)
    d = ti.min(d, p[1])
    p[1] -= 1.5

    # torso
    d = ti.min(d, p.norm() - 1.0)

    # left arm
    op = p
    p -= ti.Vector([0.66, 0.7, 0.0])
    p_xz = ti.Vector([p[0], p[2]])
    p_xz = rotmat(-0.1) @ p_xz
    p[0] = p_xz[0]
    p[2] = p_xz[1]

    d = smin(d, ((p * ti.Vector([1.0, 1.5, 1.0])) - 0.54).norm(), 0.03)
    p = op

    # mouth
    p[1] -= 0.11
    md = smax(p[2] + 0.84, smax(p[0] - 0.2, p[1] - 0.075, 0.2),
              p.dot(ti.Vector([0.7071, -0.7071, 0.0]) - 0.1, 0.08), 0.04)
    p[0] = -p[0]
    md = smax(md, smax(p[2] + 0.84, smax(p[0] - 0.2, p[1] - 0.075, 0.2),
              p.dot(ti.Vector([0.7071, -0.7071, 0.0]) - 0.1, 0.01), 0.13))
    d = smax(d, -md, 0.012)

    # tongue
    p = op
    d = smin(d, (p - ti.Vector([0.0, 0.03, -0.75])).norm() - 0.16, 0.01)

    return min(d, 10.0)
