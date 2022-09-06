# https://www.shadertoy.com/view/3l23Rh


import taichi as ti
import handy_shader_functions as hsf

ti.init(arch=ti.gpu)

res_x = 640
res_y = 360
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
m3 = ti.Vector([[0.3338, 0.56034, -0.71817], [-0.87887, 0.32651, -0.15323], [0.15162, 0.69596, 0.61339]]) * 1.93
prm1 = 0.0
bsMo = ti.Vector([0.0, 0.0])


@ti.func
def rot(a):
    c = ti.cos(a)
    s = ti.sin(a)
    mat2 = ti.Vector([[c, s], [-s, c]])
    return mat2

@ti.func
def mag2(p):
    return p.dot(p)

@ti.func
def linstep(mn, mx, x):
    return hsf.clamp((x - mn) / (mx - mn), 0.0, 1.0)

@ti.func
def disp(t):
    return ti.Vector([ti.sin(t * 0.22) * 1.0, ti.cos(t * 0.175) * 2.0])

@ti.func
def map(p, iTime):
    p2 = p
    disp_pz = disp(p[2])
    p2[0] -= disp_pz[0]
    p2[1] -= disp_pz[1]
    rot_pz = rot(ti.sin(p[2] + iTime) * (0.1 + prm1 * 0.05) + iTime * 0.09)
    p[0] *= rot_pz[0]
    p[1] *= rot_pz[1]
    p2_xy = ti.Vector([p2[0], p2[1]])

    c1 = mag2(p2_xy)
    d = 0.0
    p *= 0.61
    z = 1.0
    trk = 1.0
    dspAmp = 0.1 + prm1 * 0.2
    d_cos = ti.Vector([0.0, 0.0, 0.0])
    for i in range(5):
        p[0] += ti.sin(p[2] * 0.75 * trk + iTime * trk * 0.8) * dspAmp
        p[1] += ti.sin(p[0] * 0.75 * trk + iTime * trk * 0.8) * dspAmp
        p[2] += ti.sin(p[1] * 0.75 * trk + iTime * trk * 0.8) * dspAmp
        d_cos[0] = p[1] * z
        d_cos[1] = p[2] * z
        d_cos[2] = p[0] * z

        d -= ti.abs(ti.cos(p).dot(ti.sin(d_cos)))

        z *= 0.57
        trk *= 1.4
        p = m3 @ p

    d = abs(d + prm1 * 3.0) + prm1 * 3.0 - 2.5 + bsMo[1]
    return ti.Vector([d + c1 * 0.2 + 0.25, c1])

@ti.func
def render(ro, rd, iTime):
    rez = ti.Vector([0.0, 0.0, 0.0, 0.0])
    t = 1.5
    fogT = 0.0
    for i in range(130):
        if rez[3] > 0.99:
            break

        pos = ro + t * rd
        mpv = map(pos, iTime)
        den = hsf.clamp(mpv[0] - 0.3, 0.0, 1.0) * 1.12
        dn = hsf.clamp(mpv[0] + 2.0, 0.0, 3.0)
        col = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if mpv[0] > 0.6:
            col_sin = ti.sin(ti.Vector([5.0, 0.4, 0.2]) + mpv[1] * 0.1 + ti.sin(pos[2] * 0.4) * 0.5 + 1.8) * 0.5

            col[0] = col_sin[0]
            col[1] = col_sin[1]
            col[2] = col_sin[2]
            col[3] = 0.08
            col *= den * den * den
            linstep_col = linstep(4.0, -2.5, mpv[0]) * 2.3
            col[0] *= linstep_col
            col[1] *= linstep_col
            col[2] *= linstep_col
            pos_8 = map(pos + 0.8, iTime)
            pos_35 = map(pos + 0.35, iTime)
            dif = hsf.clamp((den - pos_8[0])/9.0, 0.001, 1.0)
            dif += hsf.clamp((den - pos_35[0])/2.5, 0.001, 1.0)
            col[0] *= den * (0.005 + 1.5 * 0.033 * dif)
            col[1] *= den * (0.045 + 1.5 * 0.07 * dif)
            col[2] *= den * (0.075 + 1.5 * 0.03 * dif)

        fogC = ti.exp(t * 0.2 - 2.2)
        col += ti.Vector([0.06, 0.11, 0.11, 0.1]) * hsf.clamp(fogC - fogT, 0.0, 1.0)
        t += hsf.clamp(0.5 - dn * dn * 0.05, 0.09, 0.3)
    return hsf.clamp(rez, 0.0, 1.0)

@ti.func
def getsat(c):
    mi = ti.min(ti.min(c[0], c[1]), c[2])
    ma = ti.max(ti.max(c[0], c[1]), c[2])
    return (ma - mi) / (ma + 1e-7)

@ti.func
def iLerp(a, b, x):
    ic = hsf.mix(a, b, x) + ti.Vector([1e-6, 0.0, 0.0])
    sd = ti.abs(getsat(ic) - hsf.mix(getsat(a), getsat(b), x))
    dir = ti.Vector([2.0*ic[0] - ic[1] - ic[2], 2.0*ic[1] - ic[0] - ic[2], 2.0*ic[2] - ic[1] - ic[0]]).normalized()
    lgt = ti.Vector([1.0, 1.0, 1.0]).dot(ic)
    ff = dir.dot(ic.normalized())
    ic += 1.5 * dir * sd * ff *lgt
    return hsf.clamp(ic, 0.0, 1.0)


@ti.kernel
def paint(iTime: ti.f32):
    for i, j in pixels:
        q = ti.Vector([float(i) / res_x, float(j) / res_y])
        p = (ti.Vector([float(i), float(j)]) - 0.5 * ti.Vector([float(res_x), float(res_y)])) / float(res_y)
        time = iTime * 3.0
        ro = ti.Vector([0.0, 0.0, time])
        ro += ti.Vector([ti.sin(iTime)*0.5, ti.sin(iTime*1.0)*0.0, 0])
        dspAmp = 0.85
        disp_roz = disp(ro[2])
        ro[0] = disp_roz[0] * dspAmp
        ro[1] = disp_roz[1] * dspAmp
        tgtDst = 3.5
        d1 = disp(time + tgtDst) * dspAmp

        target = (ro - ti.Vector([d1[0], d1[1], time + tgtDst])).normalized()
        ro[0] = bsMo[0] * 2.0
        rightdir = target.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        updir = rightdir.cross(target).normalized()
        rightdir = updir.cross(target).normalized()
        rd = ((p[0]*rightdir + p[1]*updir)*1.0 - target).normalized()
        d2 = disp(time + 3.5)
        rot2 = rot(-d2[0] * 0.2 + bsMo[0])
        rd[0] = rot2[0]
        rd[1] = rot2[1]
        prm1 = hsf.smoothstep(-0.4, 0.4, ti.sin(iTime * 0.3))
        scn = render(ro, rd, iTime)
        col = ti.Vector([scn[0], scn[1], scn[2]])
        col = iLerp(ti.Vector([col[2], col[1], col[0]]), ti.Vector([col[0], col[1], col[2]]), hsf.clamp(1.0 - prm1, 0.05, 1.0))
        col = col ** ti.Vector([0.55, 0.65, 0.6]) * ti.Vector([1.0, 0.97, 1.0])
        col *= (16.0 * q[0] * q[1] *(1.0 - q[0]) * (1.0 - q[1]))**0.12*0.7 + 0.3

        pixels[i, j] = col



gui = ti.GUI("Cloud", (res_x, res_y))
t=0.0
while True:
    t+=0.1
    # mouse = gui.get_cursor_pos()
    paint(t)

    gui.set_image(pixels)
    gui.show()