/*
 * gauss_map_viz.c  --  Edge-edge motion + signed separation visualiser
 *
 * Two reference frames A and B carry a single "box edge" each.  Each
 * frame has a start pose and an end pose; the frames translate toward
 * each other and rotate locally as t : 0 -> 1 advances.  The start
 * pose is non-colliding, the end pose is colliding.
 *
 * A 2-D plot across the bottom shows the signed edge-edge separation
 * function across t.  We use the classic scalar-triple-product form
 *
 *     sep(t) = ( (pB(t) - pA(t)) . ( dA(t) x dB(t) ) ) / | dA x dB |
 *
 * and SIGN-CORRECT by orienting the cross-product axis against the
 * vector from A's COM to B's COM, because cross(dA, dB) flips as the
 * edges rotate (non-convex over SO(3)).
 *
 * The top of the window carries a slider for scrubbing t.
 *
 * Controls
 * --------
 *   Slider drag    Scrub time t : 0..1
 *   Left-drag      Orbit camera (Maya-style, Y-up)
 *   Right-drag     Rotate the box whose centre is nearest the cursor
 *   Scroll wheel   Zoom
 *   R              Reset scene
 *   Escape         Quit
 *
 * Build (MSVC)
 *   cl gauss_map_viz.c opengl32.lib glu32.lib gdi32.lib user32.lib
 *
 * Build (MinGW-w64)
 *   gcc gauss_map_viz.c -o gauss_map_viz.exe -lopengl32 -lglu32 -lgdi32 -luser32 -mwindows
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
  #include <GLFW/glfw3.h>
  #include <GL/gl.h>
#else
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <GL/gl.h>
  #include <GL/glu.h>
  #ifdef _MSC_VER
    #pragma comment(lib, "opengl32.lib")
    #pragma comment(lib, "glu32.lib")
    #pragma comment(lib, "gdi32.lib")
    #pragma comment(lib, "user32.lib")
  #endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =================================================================
 *  Platform wrapping API
 *    - Event handlers: on_resize / on_mouse_down / on_mouse_up
 *      / on_mouse_move / on_mouse_wheel / on_key_down  (shared)
 *    - Swap:           plat_swap()                     (shared call)
 *    - Invalidate:     PLAT_INVALIDATE()               (macro)
 *  Everything else (windowing + main loop) is inside the
 *  platform-specific block at the very end of this file.
 * ================================================================= */

enum { KEY_ESCAPE = 1, KEY_R = 2, KEY_LEFT = 3, KEY_RIGHT = 4 };

#ifdef __EMSCRIPTEN__
  static GLFWwindow *g_window = NULL;
  #define PLAT_INVALIDATE() ((void)0)

  /* WebGL (even with LEGACY_GL_EMULATION) is missing a few legacy entry
     points.  Provide shims so the rest of the file compiles unchanged.  */

  static void emu_glGetDoublev(GLenum e, double *out) {
      float tmp[16];
      glGetFloatv(e, tmp);
      for (int i = 0; i < 16; i++) out[i] = (double)tmp[i];
  }
  #define glGetDoublev(e, p)  emu_glGetDoublev((e), (p))
  #define glNormal3d(x, y, z) glNormal3f((float)(x), (float)(y), (float)(z))
  #define glVertex3d(x, y, z) glVertex3f((float)(x), (float)(y), (float)(z))

  static inline void emu_glMaterialf(GLenum face, GLenum pname, float v) {
      float a[1] = { v };
      glMaterialfv(face, pname, a);
  }
  #define glMaterialf(f, p, v) emu_glMaterialf((f), (p), (v))

  /* --- Minimal inline GLU replacements (WebGL has no glu). --------- */

  static void gluPerspective(double fovy, double aspect, double zn, double zf) {
      double f = 1.0 / tan(fovy * 0.5 * M_PI / 180.0);
      float m[16] = {
          (float)(f/aspect), 0, 0, 0,
          0, (float)f, 0, 0,
          0, 0, (float)((zf+zn)/(zn-zf)), -1.0f,
          0, 0, (float)((2.0*zf*zn)/(zn-zf)), 0
      };
      glMultMatrixf(m);
  }

  static void gluLookAt(double ex, double ey, double ez,
                        double cx, double cy, double cz,
                        double ux, double uy, double uz) {
      double fx = cx-ex, fy = cy-ey, fz = cz-ez;
      double fl = sqrt(fx*fx + fy*fy + fz*fz);
      fx/=fl; fy/=fl; fz/=fl;
      double sx = fy*uz - fz*uy, sy = fz*ux - fx*uz, sz = fx*uy - fy*ux;
      double sl = sqrt(sx*sx + sy*sy + sz*sz);
      sx/=sl; sy/=sl; sz/=sl;
      double upx = sy*fz - sz*fy, upy = sz*fx - sx*fz, upz = sx*fy - sy*fx;
      float m[16] = {
          (float)sx, (float)upx, (float)-fx, 0,
          (float)sy, (float)upy, (float)-fy, 0,
          (float)sz, (float)upz, (float)-fz, 0,
          0,         0,          0,          1
      };
      glMultMatrixf(m);
      glTranslatef((float)-ex, (float)-ey, (float)-ez);
  }

  static void pm_mul_v4(const double m[16], const double v[4], double r[4]) {
      for (int i = 0; i < 4; i++)
          r[i] = m[i]*v[0] + m[i+4]*v[1] + m[i+8]*v[2] + m[i+12]*v[3];
  }

  static void pm_mul(const double a[16], const double b[16], double r[16]) {
      for (int c = 0; c < 4; c++)
          for (int ro = 0; ro < 4; ro++) {
              double s = 0;
              for (int k = 0; k < 4; k++) s += a[ro + k*4] * b[k + c*4];
              r[ro + c*4] = s;
          }
  }

  static int pm_invert(const double m[16], double inv[16]) {
      inv[0]=m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
      inv[4]=-m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
      inv[8]=m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
      inv[12]=-m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
      inv[1]=-m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
      inv[5]=m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
      inv[9]=-m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
      inv[13]=m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
      inv[2]=m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
      inv[6]=-m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
      inv[10]=m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
      inv[14]=-m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
      inv[3]=-m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
      inv[7]=m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
      inv[11]=-m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
      inv[15]=m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
      double det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
      if (det == 0) return 0;
      det = 1.0/det;
      for (int i = 0; i < 16; i++) inv[i] *= det;
      return 1;
  }

  static int gluProject(double ox, double oy, double oz,
                        const double mv[16], const double pj[16], const int vp[4],
                        double *wx, double *wy, double *wz) {
      double v[4] = {ox, oy, oz, 1.0}, t[4], c[4];
      pm_mul_v4(mv, v, t);
      pm_mul_v4(pj, t, c);
      if (c[3] == 0) return 0;
      c[0]/=c[3]; c[1]/=c[3]; c[2]/=c[3];
      *wx = vp[0] + (c[0]*0.5 + 0.5) * vp[2];
      *wy = vp[1] + (c[1]*0.5 + 0.5) * vp[3];
      *wz = c[2]*0.5 + 0.5;
      return 1;
  }

  static int gluUnProject(double wx, double wy, double wz,
                          const double mv[16], const double pj[16], const int vp[4],
                          double *ox, double *oy, double *oz) {
      double mvp[16], inv[16];
      pm_mul(pj, mv, mvp);
      if (!pm_invert(mvp, inv)) return 0;
      double v[4] = {
          (wx - vp[0]) / vp[2] * 2.0 - 1.0,
          (wy - vp[1]) / vp[3] * 2.0 - 1.0,
          wz * 2.0 - 1.0,
          1.0
      };
      double r[4];
      pm_mul_v4(inv, v, r);
      if (r[3] == 0) return 0;
      *ox = r[0]/r[3]; *oy = r[1]/r[3]; *oz = r[2]/r[3];
      return 1;
  }

  typedef struct { int dummy; } GLUquadric;
  #define GLU_FILL   0
  #define GLU_SMOOTH 0
  static GLUquadric *gluNewQuadric(void) { static GLUquadric q; return &q; }
  static void gluQuadricDrawStyle(GLUquadric *q, int s) { (void)q; (void)s; }
  static void gluQuadricNormals  (GLUquadric *q, int m) { (void)q; (void)m; }

  static void gluSphere(GLUquadric *q, double r, int slices, int stacks) {
      (void)q;
      for (int i = 0; i < stacks; i++) {
          double p0 = M_PI * ((double)i       / stacks - 0.5);
          double p1 = M_PI * ((double)(i + 1) / stacks - 0.5);
          double cp0 = cos(p0), sp0 = sin(p0);
          double cp1 = cos(p1), sp1 = sin(p1);
          glBegin(GL_QUAD_STRIP);
          for (int j = 0; j <= slices; j++) {
              double th = 2.0 * M_PI * j / slices;
              double ct = cos(th), st = sin(th);
              glNormal3d(cp1*ct, sp1, cp1*st);
              glVertex3d(r*cp1*ct, r*sp1, r*cp1*st);
              glNormal3d(cp0*ct, sp0, cp0*st);
              glVertex3d(r*cp0*ct, r*sp0, r*cp0*st);
          }
          glEnd();
      }
  }

#else
  /* Desktop Win32. */
  static HWND  g_hwnd;
  static HDC   g_hdc;
  static HGLRC g_hrc;
  #define PLAT_INVALIDATE() InvalidateRect(g_hwnd, NULL, FALSE)
#endif

static void plat_swap(void) {
#ifdef __EMSCRIPTEN__
    if (g_window) glfwSwapBuffers(g_window);
#else
    SwapBuffers(g_hdc);
#endif
}

/* =================================================================
 *  Vec3
 * ================================================================= */

typedef struct { float x, y, z; } Vec3;

static inline Vec3  v3(float x, float y, float z) { return (Vec3){x, y, z}; }
static inline Vec3  v_add(Vec3 a, Vec3 b)   { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3  v_sub(Vec3 a, Vec3 b)   { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3  v_scale(Vec3 v, float s){ return v3(v.x*s, v.y*s, v.z*s); }
static inline Vec3  v_neg(Vec3 v)           { return v3(-v.x, -v.y, -v.z); }
static inline float v_dot(Vec3 a, Vec3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3  v_cross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x);
}
static inline float v_len(Vec3 v) { return sqrtf(v_dot(v, v)); }
static inline Vec3  v_normalize(Vec3 v) {
    float l = v_len(v);
    return l > 1e-8f ? v_scale(v, 1.0f / l) : v3(0, 0, 0);
}
static inline Vec3  v_lerp(Vec3 a, Vec3 b, float t) {
    return v3(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t);
}

/* Spherical linear interpolation between two unit vectors. */
static Vec3 v_slerp(Vec3 a, Vec3 b, float t) {
    float d = v_dot(a, b);
    if (d > 0.9999f)
        return v_normalize(v_add(v_scale(a, 1.0f - t), v_scale(b, t)));
    if (d < -0.9999f) {
        Vec3 perp = v3(0, 1, 0);
        if (fabsf(v_dot(a, perp)) > 0.9f) perp = v3(1, 0, 0);
        perp = v_normalize(v_cross(a, perp));
        float th = (float)M_PI * t;
        return v_normalize(v_add(v_scale(a, cosf(th)),
                                 v_scale(perp, sinf(th))));
    }
    float th = acosf(d), st = sinf(th);
    float wa = sinf((1.0f - t) * th) / st;
    float wb = sinf(t * th) / st;
    return v3(wa*a.x + wb*b.x, wa*a.y + wb*b.y, wa*a.z + wb*b.z);
}

/* =================================================================
 *  Quaternion
 * ================================================================= */

typedef struct { float x, y, z, w; } Quat;

static inline Quat q_ident(void) { return (Quat){0.0f, 0.0f, 0.0f, 1.0f}; }

static Quat q_from_axis_angle(Vec3 axis, float angle) {
    Vec3 a = v_normalize(axis);
    float s = sinf(angle * 0.5f);
    return (Quat){ a.x*s, a.y*s, a.z*s, cosf(angle * 0.5f) };
}

static Quat q_mul(Quat a, Quat b) {
    return (Quat){
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    };
}

static Quat q_normalize(Quat q) {
    float l = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (l < 1e-8f) return q_ident();
    float s = 1.0f / l;
    return (Quat){ q.x*s, q.y*s, q.z*s, q.w*s };
}

static Vec3 q_rotate(Quat q, Vec3 v) {
    Vec3 qv  = v3(q.x, q.y, q.z);
    Vec3 t   = v_scale(v_cross(qv, v), 2.0f);
    return v_add(v, v_add(v_scale(t, q.w), v_cross(qv, t)));
}

static inline float q_dot(Quat a, Quat b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

static inline Quat q_neg(Quat q) { return (Quat){ -q.x, -q.y, -q.z, -q.w }; }

/* Shortest-arc rotation quaternion taking unit v0 to unit v1.
   Direct port of Game Programming Gems' RotationArc -- callers must
   supply unit-length inputs.                                       */
static Quat q_rotation_arc(Vec3 v0, Vec3 v1) {
    Vec3  c = v_cross(v0, v1);
    float d = v_dot(v0, v1);
    if (d <= -1.0f) return (Quat){ 1.0f, 0.0f, 0.0f, 0.0f }; /* 180 about x */
    float s = sqrtf((1.0f + d) * 2.0f);
    return (Quat){ c.x / s, c.y / s, c.z / s, s * 0.5f };
}

static Quat q_slerp(Quat a, Quat b, float t) {
    float d = q_dot(a, b);
    if (d < 0.0f) { b = q_neg(b); d = -d; }
    if (d > 0.9995f) {
        Quat r = { a.x + t*(b.x - a.x),
                   a.y + t*(b.y - a.y),
                   a.z + t*(b.z - a.z),
                   a.w + t*(b.w - a.w) };
        return q_normalize(r);
    }
    float th = acosf(d);
    float st = sinf(th);
    float s1 = sinf((1.0f - t) * th) / st;
    float s2 = sinf(t * th) / st;
    return (Quat){
        a.x*s1 + b.x*s2,
        a.y*s1 + b.y*s2,
        a.z*s1 + b.z*s2,
        a.w*s1 + b.w*s2
    };
}

/* =================================================================
 *  Box (reference frame carrying an edge)
 * ================================================================= */

typedef struct {
    Vec3 size;                 /* half-extents                         */
    Vec3 pos_start, pos_end;
    Quat rot_start, rot_end;

    /* Edge in local box space.  For a box polyhedron, an edge is the
       intersection of two face-planes whose normals are orthogonal.   */
    Vec3 edge_center_local;    /* midpoint of the edge in local coords */
    Vec3 edge_dir_local;       /* unit direction of the edge           */
    float edge_half_len;
    Vec3 face_n1_local;        /* unit normal of one adjacent face     */
    Vec3 face_n2_local;        /* unit normal of the other             */

    /* Motion parameters driving rot_end = R(axis, angle) * rot_start.
       A vertical slider on the LHS modulates the angle; the axis stays
       fixed in the WORLD frame (doesn't follow the box through the
       motion).  Refresh rot_end with box_refresh_motion after any
       change to motion_angle or rot_start.                            */
    Vec3  motion_axis_world;
    float motion_angle;
} Box;

static Box   box_a, box_b;
static float t_cur = 0.0f;     /* current time in [0, 1]               */

static void box_refresh_motion(Box *b) {
    b->rot_end = q_normalize(q_mul(
        q_from_axis_angle(b->motion_axis_world, b->motion_angle),
        b->rot_start));
}

/* Per-box pose at time t. */
static void box_pose_at(const Box *b, float t, Vec3 *pos_out, Quat *rot_out) {
    *pos_out = v_lerp(b->pos_start, b->pos_end, t);
    *rot_out = q_slerp(b->rot_start, b->rot_end, t);
}

/* Transform a local-space point by the box's pose at time t. */
static Vec3 box_point_world(const Box *b, float t, Vec3 local) {
    Vec3 p; Quat r;
    box_pose_at(b, t, &p, &r);
    return v_add(p, q_rotate(r, local));
}

static Vec3 box_dir_world(const Box *b, float t, Vec3 local) {
    Vec3 p; Quat r;
    box_pose_at(b, t, &p, &r);
    return q_rotate(r, local);
}

/* World-space edge endpoints at time t. */
static void box_edge_world(const Box *b, float t, Vec3 *p0, Vec3 *p1) {
    Vec3 p; Quat r;
    box_pose_at(b, t, &p, &r);
    Vec3 c = v_add(p, q_rotate(r, b->edge_center_local));
    Vec3 d = q_rotate(r, b->edge_dir_local);
    *p0 = v_add(c, v_scale(d, -b->edge_half_len));
    *p1 = v_add(c, v_scale(d,  b->edge_half_len));
}

/* =================================================================
 *  Separation functions  (swap these while prototyping)
 * ================================================================= */

/* Dirk Gregorius' edge-axis sign correction (lmQueryEdgeAxes,
   GDC 2013):  flip n so that  dot(n, PA - CA) > 0,  where PA is one
   vertex of edge A and CA is box A's centroid.  Geometrically this
   orients n outward from A's body through edge A -- the same intent
   as the outward-bisector correction, but weighted by A's half-
   extents rather than uniformly.                                   */
static float sep_edge_edge_signed(const Box *A, const Box *B, float t) {
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(A, t, &pA0, &pA1);
    box_edge_world(B, t, &pB0, &pB1);

    Vec3 pA = v_scale(v_add(pA0, pA1), 0.5f);
    Vec3 pB = v_scale(v_add(pB0, pB1), 0.5f);
    Vec3 dA = v_normalize(v_sub(pA1, pA0));
    Vec3 dB = v_normalize(v_sub(pB1, pB0));

    Vec3  n  = v_cross(dA, dB);
    float nl = v_len(n);
    if (nl < 1e-4f) return 9999.0f;

    Vec3 n_hat = v_scale(n, 1.0f / nl);

    Vec3 comA_t; Quat rA;
    box_pose_at(A, t, &comA_t, &rA);
    if (v_dot(n_hat, v_sub(pA0, comA_t)) < 0.0f)
        n_hat = v_neg(n_hat);

    return v_dot(n_hat, v_sub(pB, pA));
}

/* Outward-based sign correction: cross axis flipped against box A's
   edge-outward direction, which is the bisector of its two adjacent
   face normals  (n_a1 + n_a2).  That direction is rigidly attached to
   the box and rotates smoothly with it.  When the Minkowski face is
   valid, cross(dA, dB) lies in the arc between n_a1 and n_a2, so the
   dot product stays stably positive -- no spurious sign flip.       */
static float sep_edge_edge_outward(const Box *A, const Box *B, float t) {
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(A, t, &pA0, &pA1);
    box_edge_world(B, t, &pB0, &pB1);

    Vec3 pA = v_scale(v_add(pA0, pA1), 0.5f);
    Vec3 pB = v_scale(v_add(pB0, pB1), 0.5f);
    Vec3 dA = v_normalize(v_sub(pA1, pA0));
    Vec3 dB = v_normalize(v_sub(pB1, pB0));

    Vec3  n  = v_cross(dA, dB);
    float nl = v_len(n);
    if (nl < 1e-4f) return 9999.0f;

    Vec3 n_hat = v_scale(n, 1.0f / nl);

    Vec3 nA1 = box_dir_world(A, t, A->face_n1_local);
    Vec3 nA2 = box_dir_world(A, t, A->face_n2_local);
    Vec3 outward = v_normalize(v_add(nA1, nA2));
    if (v_dot(n_hat, outward) < 0.0f)
        n_hat = v_neg(n_hat);

    return v_dot(n_hat, v_sub(pB, pA));
}

/* Raw unnormalised signed scalar triple product.  Polynomial in the
   motion parameters -- no |cross| denominator, no conditional sign
   flip, no branches anywhere.  The function is smooth (C-infinity
   in the motion parameters) and free of the rotational-parity flips
   that produce false roots in CCD.  Sign is the chirality of
   (vB - vA, EA, EB); fix the convention externally once (e.g. at a
   known-separated reference frame) instead of per-query.          */
static float sep_edge_edge_triple(const Box *A, const Box *B, float t) {
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(A, t, &pA0, &pA1);
    box_edge_world(B, t, &pB0, &pB1);
    Vec3 EA = v_sub(pA1, pA0);
    Vec3 EB = v_sub(pB1, pB0);
    return v_dot(v_sub(pB0, pA0), v_cross(EA, EB));
}

/* Regularised signed line-line distance:
       sep = sgn * (vB - vA) . (EA x EB) / sqrt(|EA x EB|^2 + eps)
   Equals the true signed distance when |cross|^2 >> eps; smoothly
   collapses toward zero through the parallel-edge singularity
   instead of spiking.  Sign is corrected outward from A so that the
   convention matches the green curve (positive = separated).      */
static float sep_edge_edge_regularised(const Box *A, const Box *B, float t) {
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(A, t, &pA0, &pA1);
    box_edge_world(B, t, &pB0, &pB1);
    Vec3 EA = v_sub(pA1, pA0);
    Vec3 EB = v_sub(pB1, pB0);
    Vec3 c  = v_cross(EA, EB);

    Vec3 nA1 = box_dir_world(A, t, A->face_n1_local);
    Vec3 nA2 = box_dir_world(A, t, A->face_n2_local);
    Vec3 outward = v_add(nA1, nA2);
    float sgn = (v_dot(c, outward) < 0.0f) ? -1.0f : 1.0f;

    const float eps = 0.05f;
    return sgn * v_dot(v_sub(pB0, pA0), c) / sqrtf(v_dot(c, c) + eps);
}

/* Closest-approach parameters (s, t) between two infinite lines.
   Lines are P_A = pA + s * dA  and  P_B = pB + t * dB  (dA, dB unit).
   `s_out`, `t_out` are the line-parameters of the common perpendicular
   foot on each line; if they fall inside each edge's half-length the
   contact is a true edge-edge contact, not a vertex/face contact.    */
static bool line_closest_params(Vec3 pA, Vec3 dA,
                                 Vec3 pB, Vec3 dB,
                                 float *s_out, float *t_out) {
    Vec3  w     = v_sub(pA, pB);
    float b     = v_dot(dA, dB);
    float denom = 1.0f - b * b;
    if (fabsf(denom) < 1e-6f) return false;       /* parallel */
    float d = v_dot(dA, w);
    float e = v_dot(dB, w);
    *t_out = -(e + b * d) / denom;
    *s_out = -d + (*t_out) * b;
    return true;
}

/* Unsigned (raw) scalar triple product normalised by |dA x dB|.
   Kept so you can A/B the sign-corrected version against the raw
   (non-convex) one on the plot.                                     */
static float sep_edge_edge_raw(const Box *A, const Box *B, float t) {
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(A, t, &pA0, &pA1);
    box_edge_world(B, t, &pB0, &pB1);

    Vec3 pA = v_scale(v_add(pA0, pA1), 0.5f);
    Vec3 pB = v_scale(v_add(pB0, pB1), 0.5f);
    Vec3 dA = v_normalize(v_sub(pA1, pA0));
    Vec3 dB = v_normalize(v_sub(pB1, pB0));

    Vec3  n  = v_cross(dA, dB);
    float nl = v_len(n);
    if (nl < 1e-4f) return 9999.0f;

    return v_dot(v_sub(pB, pA), n) / nl;
}

/* =================================================================
 *  Minkowski face test (from the original file; still useful to flag
 *  when the current axis is a valid SAT axis).
 * ================================================================= */

/* Minkowski-face test (Dirk Gregorius, GDC 2013).  Returns true iff
   the great-circle arc a->b on the unit sphere strictly crosses arc
   c->d.  For box-box edge SAT, callers must pass (UA, VA, -UB, -VB)
   so that the test reads "does arc(UA,VA) cross arc(-UB,-VB)" -- the
   condition for cross(EA, EB) to be a valid Minkowski-difference
   face.

   Conditions:
     CBA * DBA < 0   c, d on opposite sides of plane(a, b)
     ADC * BDC < 0   a, b on opposite sides of plane(c, d)
     CBA * BDC > 0   crossing is in the same hemisphere (filters out
                     antipodal great-circle crossings that don't lie
                     inside both short arcs)                          */
static inline bool is_minkowski_face(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    Vec3  bxa = v_cross(b, a);
    Vec3  dxc = v_cross(d, c);
    float CBA = v_dot(c, bxa);
    float DBA = v_dot(d, bxa);
    float ADC = v_dot(a, dxc);
    float BDC = v_dot(b, dxc);
    return (CBA * DBA < 0.0f) &
           (ADC * BDC < 0.0f) &
           (CBA * BDC > 0.0f);
}

/* Face normals of a box's edge, expressed in world space at time t.  */
static void box_edge_face_normals_world(const Box *b, float t,
                                         Vec3 *n1, Vec3 *n2) {
    *n1 = box_dir_world(b, t, b->face_n1_local);
    *n2 = box_dir_world(b, t, b->face_n2_local);
}

/* =================================================================
 *  Application state
 * ================================================================= */

static int   win_w = 1200, win_h = 820;

/* UI regions. */
#define SLIDER_H    44
#define PLOT_H      200
#define PLOT_PAD    30
#define SIDEBAR_W   78
#define ROT_MAX     3.0f          /* max motion-rotation slider value (rad) */

/* Gauss-map inset geometry (same values used for drawing and hit-testing). */
#define INSET_W      260
#define INSET_H      260
#define INSET_PAD    14
#define INSET_TITLE  22

/* Main 3-D camera.  Orbits around (SCENE_SHIFT_X, 0, 0) -- the scene is
   offset left to leave room for the Gauss-map inset on the right.      */
static float SCENE_SHIFT_X = 0.9f;
static float cam_az   =  0.7f;
static float cam_el   =  0.35f;
static float cam_dist =  8.0f;

/* Gauss-map inset camera (independent orbit). */
static float cam_az_g   =  0.7f;
static float cam_el_g   =  0.35f;
static float cam_dist_g =  3.0f;

/* Interaction. */
static bool  lmb_down = false, rmb_down = false;
static bool  slider_drag = false;
static bool  inset_drag = false;       /* LMB orbiting the inset camera   */
static int   rot_slider_drag = -1;     /* 0 = A slider, 1 = B slider, -1 = none */
static int   mx_prev, my_prev;
static int   box_drag_idx = -1;         /* 0 = A, 1 = B, -1 = none        */

/* Bitmap font. */
static GLuint font_base;

/* =================================================================
 *  Scene setup
 * ================================================================= */

/* Initial poses are chosen so that the edge-pair's Gauss-map arcs
 * (a -> b and -c -> -d) cross at BOTH t=0 and t=1.  Box A stays axis-
 * aligned; box B carries a 180-degree rotation about (1/sqrt 2, -1/2,
 * 1/2) which sends its top-front edge's face normals (0,1,0), (0,0,1)
 * to ( -1/sqrt2, -1/2, -1/2 ) and ( +1/sqrt2, -1/2, -1/2 ).  Negated
 * for the Minkowski sum, those land symmetrically about the y=z plane
 * and cross the arc (0,1,0) -> (0,0,1) cleanly at its midpoint.  Small
 * Y-axis rotations on both boxes between t=0 and t=1 give visible
 * rotational motion while staying well inside the Minkowski region.  */
static void reset_scene(void) {
    t_cur = 0.0f;

    /* --- Box A: axis-aligned.  Top-front edge (local +Y, +Z faces). */
    box_a.size               = v3(0.9f, 0.7f, 0.8f);
    box_a.pos_start          = v3(-3.0f, 0.0f, 0.0f);
    box_a.pos_end            = v3(-0.3f, 0.0f, 0.0f);
    box_a.rot_start          = q_ident();
    box_a.edge_center_local  = v3(0.0f, 0.7f, 0.8f);
    box_a.edge_dir_local     = v3(1.0f, 0.0f, 0.0f);
    box_a.edge_half_len      = 0.9f;
    box_a.face_n1_local      = v3(0.0f, 1.0f, 0.0f);
    box_a.face_n2_local      = v3(0.0f, 0.0f, 1.0f);
    box_a.motion_axis_world  = v3(0.0f, 1.0f, 0.0f);
    box_a.motion_angle       = 0.7f;
    box_refresh_motion(&box_a);

    /* --- Box B: 180-degree rotation places its top-front edge so its
       face normals are symmetric about x=0, giving crossing arcs.     */
    const float s = 0.70710678f;              /* 1 / sqrt(2)           */
    Vec3 axis_b = v3(s, -0.5f, 0.5f);         /* already unit          */
    Quat r180  = q_from_axis_angle(axis_b, (float)M_PI);

    box_b.size               = v3(0.9f, 0.7f, 0.8f);
    /* Approach vector has a large (y+z) component so the signed
       separation starts positive and sweeps through zero into
       negative (penetrating) as t advances.                           */
    box_b.pos_start          = v3( 3.0f, 2.2f, 2.2f);
    box_b.pos_end            = v3( 0.20f, 1.15f, 1.25f);
    box_b.rot_start          = r180;
    box_b.edge_center_local  = v3(0.0f, 0.7f, 0.8f);
    box_b.edge_dir_local     = v3(1.0f, 0.0f, 0.0f);
    box_b.edge_half_len      = 0.9f;
    box_b.face_n1_local      = v3(0.0f, 1.0f, 0.0f);
    box_b.face_n2_local      = v3(0.0f, 0.0f, 1.0f);
    box_b.motion_axis_world  = v3(0.0f, 1.0f, 0.0f);
    box_b.motion_angle       = 1.0f;
    box_refresh_motion(&box_b);
}

/* =================================================================
 *  Font
 * ================================================================= */

#ifdef __EMSCRIPTEN__
/* Text rendering is stubbed on web -- labels won't appear, geometry still works. */
static void init_font(void) {}
static void gl_text_2d(int x, int y, const char *s) { (void)x; (void)y; (void)s; }
static void gl_text_3d(Vec3 p, const char *s) { (void)p; (void)s; }
#else
static void init_font(void) {
    font_base = glGenLists(128);
    HFONT f = CreateFontA(
        -14, 0, 0, 0, FW_BOLD, 0, 0, 0,
        ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH, "Consolas");
    HFONT old = (HFONT)SelectObject(g_hdc, f);
    wglUseFontBitmaps(g_hdc, 0, 128, font_base);
    SelectObject(g_hdc, old);
    DeleteObject(f);
}

/* x from left, y from top. */
static void gl_text_2d(int x, int y, const char *s) {
    glRasterPos2i(x, win_h - y);
    glListBase(font_base);
    glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, (const GLubyte *)s);
}

/* Draws text at a 3-D position in the current model-view / projection. */
static void gl_text_3d(Vec3 p, const char *s) {
    glRasterPos3f(p.x, p.y, p.z);
    glListBase(font_base);
    glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, (const GLubyte *)s);
}
#endif

/* =================================================================
 *  3-D drawing
 * ================================================================= */

/* Draw a single box (wireframe + translucent faces) using its pose
   at time t.  Highlights the "active" edge in a bright colour.      */
static void draw_box(const Box *b, float t,
                     float cr, float cg, float cb, float edge_r,
                     float edge_g, float edge_b) {
    Vec3 p; Quat r;
    box_pose_at(b, t, &p, &r);

    /* Box corners in local coords. */
    Vec3 hx = v3(b->size.x, 0, 0);
    Vec3 hy = v3(0, b->size.y, 0);
    Vec3 hz = v3(0, 0, b->size.z);
    Vec3 corner[8];
    for (int i = 0; i < 8; i++) {
        Vec3 l = v_add(v_add(
            v_scale(hx, (i & 1) ? 1.0f : -1.0f),
            v_scale(hy, (i & 2) ? 1.0f : -1.0f)),
            v_scale(hz, (i & 4) ? 1.0f : -1.0f));
        corner[i] = v_add(p, q_rotate(r, l));
    }

    static const int edges[12][2] = {
        {0,1},{2,3},{4,5},{6,7},
        {0,2},{1,3},{4,6},{5,7},
        {0,4},{1,5},{2,6},{3,7},
    };
    static const int faces[6][4] = {
        {0,2,6,4}, {1,3,7,5},
        {0,1,5,4}, {2,3,7,6},
        {0,1,3,2}, {4,5,7,6},
    };

    /* Translucent faces. */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(cr, cg, cb, 0.12f);
    glBegin(GL_QUADS);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 4; j++) {
            Vec3 v = corner[faces[i][j]];
            glVertex3f(v.x, v.y, v.z);
        }
    }
    glEnd();

    /* Wire edges. */
    glLineWidth(1.8f);
    glColor3f(cr * 0.85f, cg * 0.85f, cb * 0.85f);
    glBegin(GL_LINES);
    for (int i = 0; i < 12; i++) {
        Vec3 a = corner[edges[i][0]];
        Vec3 b2 = corner[edges[i][1]];
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b2.x, b2.y, b2.z);
    }
    glEnd();

    /* Active edge highlight. */
    Vec3 e0, e1;
    box_edge_world(b, t, &e0, &e1);
    glLineWidth(5.0f);
    glColor3f(edge_r, edge_g, edge_b);
    glBegin(GL_LINES);
    glVertex3f(e0.x, e0.y, e0.z);
    glVertex3f(e1.x, e1.y, e1.z);
    glEnd();

    /* Edge endpoint dots. */
    glPointSize(9.0f);
    glBegin(GL_POINTS);
    glVertex3f(e0.x, e0.y, e0.z);
    glVertex3f(e1.x, e1.y, e1.z);
    glEnd();

    /* Face normals at the edge midpoint (short debug sticks). */
    Vec3 emid = v_scale(v_add(e0, e1), 0.5f);
    Vec3 n1, n2;
    box_edge_face_normals_world(b, t, &n1, &n2);
    glLineWidth(1.5f);
    glColor3f(cr * 0.6f, cg * 0.6f, cb * 0.6f);
    glBegin(GL_LINES);
    glVertex3f(emid.x, emid.y, emid.z);
    Vec3 m1 = v_add(emid, v_scale(n1, 0.35f));
    glVertex3f(m1.x, m1.y, m1.z);
    glVertex3f(emid.x, emid.y, emid.z);
    Vec3 m2 = v_add(emid, v_scale(n2, 0.35f));
    glVertex3f(m2.x, m2.y, m2.z);
    glEnd();

    glDisable(GL_BLEND);
}

/* Faint ghost of the start/end poses. */
static void draw_ghost_box(const Box *b, float t_g, float rr, float gg, float bb) {
    Vec3 p; Quat r;
    box_pose_at(b, t_g, &p, &r);

    Vec3 hx = v3(b->size.x, 0, 0);
    Vec3 hy = v3(0, b->size.y, 0);
    Vec3 hz = v3(0, 0, b->size.z);
    Vec3 corner[8];
    for (int i = 0; i < 8; i++) {
        Vec3 l = v_add(v_add(
            v_scale(hx, (i & 1) ? 1.0f : -1.0f),
            v_scale(hy, (i & 2) ? 1.0f : -1.0f)),
            v_scale(hz, (i & 4) ? 1.0f : -1.0f));
        corner[i] = v_add(p, q_rotate(r, l));
    }
    static const int edges[12][2] = {
        {0,1},{2,3},{4,5},{6,7},
        {0,2},{1,3},{4,6},{5,7},
        {0,4},{1,5},{2,6},{3,7},
    };
    glLineWidth(1.0f);
    glColor4f(rr, gg, bb, 0.35f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINES);
    for (int i = 0; i < 12; i++) {
        Vec3 a = corner[edges[i][0]];
        Vec3 b2 = corner[edges[i][1]];
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b2.x, b2.y, b2.z);
    }
    glEnd();

    /* Ghost edge. */
    Vec3 e0, e1;
    box_edge_world(b, t_g, &e0, &e1);
    glLineWidth(2.0f);
    glColor4f(rr, gg, bb, 0.5f);
    glBegin(GL_LINES);
    glVertex3f(e0.x, e0.y, e0.z);
    glVertex3f(e1.x, e1.y, e1.z);
    glEnd();

    glDisable(GL_BLEND);
}

/* Ground grid. */
static void draw_ground(void) {
    glColor4f(0.28f, 0.28f, 0.33f, 0.8f);
    glLineWidth(1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINES);
    float y = -1.6f;
    for (int i = -8; i <= 8; i++) {
        glVertex3f((float)i, y, -8.0f);
        glVertex3f((float)i, y,  8.0f);
        glVertex3f(-8.0f, y, (float)i);
        glVertex3f( 8.0f, y, (float)i);
    }
    glEnd();
    glDisable(GL_BLEND);
}

/* =================================================================
 *  Camera
 * ================================================================= */

/* Make 3-D rendering use the middle strip, minus the LHS sidebar.     */
static int view3d_x0(void) { return SIDEBAR_W; }
static int view3d_y0(void) { return PLOT_H; }                 /* GL origin = bottom-left */
static int view3d_w (void) { return win_w - SIDEBAR_W; }
static int view3d_h (void) { return win_h - SLIDER_H - PLOT_H; }

static void setup_camera_3d(void) {
    glViewport(view3d_x0(), view3d_y0(), view3d_w(), view3d_h());

    float cx = SCENE_SHIFT_X + cam_dist * cosf(cam_el) * sinf(cam_az);
    float cy = cam_dist * sinf(cam_el);
    float cz = cam_dist * cosf(cam_el) * cosf(cam_az);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0,
                   (double)view3d_w() / (double)view3d_h(),
                   0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cx, cy, cz, SCENE_SHIFT_X, 0, 0, 0, 1, 0);
}

/* Camera position in world for the main 3-D view. */
static Vec3 camera_position_world(void) {
    return v3(SCENE_SHIFT_X + cam_dist * cosf(cam_el) * sinf(cam_az),
              cam_dist * sinf(cam_el),
              cam_dist * cosf(cam_el) * cosf(cam_az));
}

/* Unproject a window-pixel mouse coordinate (y from top) into a world-
   space ray starting at the near plane and going through the pixel.    */
static void screen_to_ray(int mx, int my_top, Vec3 *ro, Vec3 *rd) {
    setup_camera_3d();
    GLdouble mv[16], pj[16];
    GLint    vp[4];
    double   ox, oy, oz, fx, fy, fz;

    glGetDoublev(GL_MODELVIEW_MATRIX,  mv);
    glGetDoublev(GL_PROJECTION_MATRIX, pj);
    glGetIntegerv(GL_VIEWPORT, vp);

    double mx_d = (double)mx;
    double my_d = (double)(win_h - my_top - 1);

    gluUnProject(mx_d, my_d, 0.0, mv, pj, vp, &ox, &oy, &oz);
    gluUnProject(mx_d, my_d, 1.0, mv, pj, vp, &fx, &fy, &fz);

    *ro = v3((float)ox, (float)oy, (float)oz);
    *rd = v_normalize(v3((float)(fx - ox), (float)(fy - oy), (float)(fz - oz)));
}

/* Virtual-trackball rotation (Gems-style) around a world-space centre.
   Picks the rotation that carries the sphere-intersection of ray 1 to
   the sphere-intersection of ray 2, with a fudge factor scaling the
   trackball radius with the distance between COP and COR.              */
static Quat virtual_trackball(Vec3 cop, Vec3 cor, Vec3 dir1, Vec3 dir2) {
    Vec3  nrml  = v_sub(cor, cop);
    float nlen  = v_len(nrml);
    if (nlen < 1e-6f) return q_ident();
    float fudge = 1.0f / (nlen * 0.25f);
    nrml = v_scale(nrml, 1.0f / nlen);
    float dist = -v_dot(nrml, cor);

    float t_cop = v_dot(nrml, cop) + dist;

    float denom1 = v_dot(nrml, dir1);
    if (fabsf(denom1) < 1e-6f) return q_ident();
    float t1 = -t_cop / denom1;
    Vec3 up  = v_sub(v_add(cop, v_scale(dir1, t1)), cor);
    up = v_scale(up, fudge);
    float um = v_len(up);
    if (um > 1.0f)  up = v_scale(up, 1.0f / um);
    else            up = v_sub(up, v_scale(nrml, sqrtf(1.0f - um * um)));

    float denom2 = v_dot(nrml, dir2);
    if (fabsf(denom2) < 1e-6f) return q_ident();
    float t2 = -t_cop / denom2;
    Vec3 vp = v_sub(v_add(cop, v_scale(dir2, t2)), cor);
    vp = v_scale(vp, fudge);
    float vm = v_len(vp);
    if (vm > 1.0f)  vp = v_scale(vp, 1.0f / vm);
    else            vp = v_sub(vp, v_scale(nrml, sqrtf(1.0f - vm * vm)));

    return q_rotation_arc(v_normalize(up), v_normalize(vp));
}

static void setup_ortho_full(void) {
    glViewport(0, 0, win_w, win_h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, win_w, 0, win_h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

/* =================================================================
 *  Picking
 * ================================================================= */

/* Project a world-space point to screen coordinates using the current
   3-D view settings.  Returns false if behind the camera.            */
static bool project_world(Vec3 p, float *sx, float *sy) {
    GLdouble mv[16], pj[16];
    GLint    vp[4];
    double   wx, wy, wz;

    glGetDoublev(GL_MODELVIEW_MATRIX,  mv);
    glGetDoublev(GL_PROJECTION_MATRIX, pj);
    glGetIntegerv(GL_VIEWPORT, vp);

    if (gluProject(p.x, p.y, p.z, mv, pj, vp, &wx, &wy, &wz)) {
        *sx = (float)wx;
        *sy = (float)wy;      /* GL origin = bottom-left */
        return true;
    }
    return false;
}

/* Pick the nearest box centre on screen (returns 0, 1, or -1). */
static int pick_box(int mx, int my_top) {
    setup_camera_3d();
    float my_gl = (float)(win_h - my_top - 1);  /* to GL bottom-left */

    Vec3 ca, cb;
    Quat ra, rb;
    box_pose_at(&box_a, t_cur, &ca, &ra);
    box_pose_at(&box_b, t_cur, &cb, &rb);

    float sxA, syA, sxB, syB;
    bool okA = project_world(ca, &sxA, &syA);
    bool okB = project_world(cb, &sxB, &syB);

    int best = -1;
    float best_d2 = 1e18f;
    if (okA) {
        float dx = sxA - mx, dy = syA - my_gl;
        float d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best = 0; }
    }
    if (okB) {
        float dx = sxB - mx, dy = syB - my_gl;
        float d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best = 1; }
    }
    return best;
}

/* =================================================================
 *  Slider / plot geometry
 * ================================================================= */

/* All using (x-from-left, y-from-top) pixel coords. */

/* Slider track. */
static int slider_x0(void) { return SIDEBAR_W + 80; }
static int slider_x1(void) { return win_w - 40; }
static int slider_y (void) { return SLIDER_H / 2; }

/* --- LHS sidebar: two vertical sliders controlling box motion_angle. */

/* Sidebar occupies x in [0, SIDEBAR_W] and the same y-range as the
   3-D view.  Each rotation slider is centred in one half of the
   sidebar (idx=0 -> box A on the left, idx=1 -> box B on the right). */
static int rot_track_x(int idx) { return SIDEBAR_W / 4 + idx * (SIDEBAR_W / 2); }
static int rot_track_y_top(void) { return SLIDER_H + 38; }
static int rot_track_y_bot(void) { return win_h - PLOT_H - 40; }

static int rot_knob_y(int idx) {
    float v = (idx == 0 ? box_a.motion_angle : box_b.motion_angle);
    if (v < 0) v = 0;
    if (v > ROT_MAX) v = ROT_MAX;
    int top = rot_track_y_top(), bot = rot_track_y_bot();
    return bot - (int)((bot - top) * (v / ROT_MAX) + 0.5f);
}

static bool rot_slider_hit(int mx, int my, int idx) {
    int tx  = rot_track_x(idx);
    int top = rot_track_y_top(), bot = rot_track_y_bot();
    return mx >= tx - 14 && mx <= tx + 14 &&
           my >= top - 10 && my <= bot + 10;
}

static void rot_slider_update(int my, int idx) {
    int top = rot_track_y_top(), bot = rot_track_y_bot();
    float v = (float)(bot - my) / (float)(bot - top) * ROT_MAX;
    if (v < 0) v = 0;
    if (v > ROT_MAX) v = ROT_MAX;
    Box *b = (idx == 0) ? &box_a : &box_b;
    b->motion_angle = v;
    box_refresh_motion(b);
}

static int slider_knob_x(void) {
    return slider_x0() + (int)((slider_x1() - slider_x0()) * t_cur + 0.5f);
}

static bool slider_hit(int mx, int my) {
    if (my > SLIDER_H + 6) return false;
    int dy = my - slider_y();
    if (dy < -12 || dy > 12) return false;
    return mx >= slider_x0() - 8 && mx <= slider_x1() + 8;
}

static void slider_update(int mx) {
    float w = (float)(slider_x1() - slider_x0());
    float v = ((float)mx - slider_x0()) / w;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    t_cur = v;
}

/* Plot area (bottom strip). */
static int plot_x0(void) { return 60; }
static int plot_x1(void) { return win_w - 20; }
static int plot_y_top(void) { return win_h - PLOT_H + 14; }   /* y-from-top */
static int plot_y_bot(void) { return win_h - 14; }

static bool plot_hit(int my) { return my >= win_h - PLOT_H; }

/* =================================================================
 *  Slider / plot drawing
 * ================================================================= */

/* Filled rect in 2-D overlay coords (x,y from top-left, in px). */
static void fill_rect_2d(int x, int y, int w, int h,
                         float r, float g, float b, float a) {
    int y_gl_bot = win_h - (y + h);
    int y_gl_top = win_h - y;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2i(x,     y_gl_bot);
    glVertex2i(x + w, y_gl_bot);
    glVertex2i(x + w, y_gl_top);
    glVertex2i(x,     y_gl_top);
    glEnd();
    glDisable(GL_BLEND);
}

static void line_2d(float x0, float y0, float x1, float y1,
                    float r, float g, float b, float lw) {
    glLineWidth(lw);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex2f(x0, (float)win_h - y0);
    glVertex2f(x1, (float)win_h - y1);
    glEnd();
}

static void draw_slider(void) {
    /* Background. */
    fill_rect_2d(0, 0, win_w, SLIDER_H, 0.13f, 0.13f, 0.17f, 1.0f);

    /* Track. */
    int x0 = slider_x0(), x1 = slider_x1(), y = slider_y();
    fill_rect_2d(x0, y - 2, x1 - x0, 4, 0.40f, 0.40f, 0.48f, 1.0f);

    /* Filled portion up to knob. */
    int kx = slider_knob_x();
    fill_rect_2d(x0, y - 2, kx - x0, 4, 0.35f, 0.70f, 1.00f, 1.0f);

    /* End-caps 0 and 1. */
    fill_rect_2d(x0 - 1, y - 7, 2, 14, 0.65f, 0.65f, 0.70f, 1.0f);
    fill_rect_2d(x1 - 1, y - 7, 2, 14, 0.65f, 0.65f, 0.70f, 1.0f);

    /* Knob. */
    fill_rect_2d(kx - 6, y - 10, 12, 20, 0.95f, 0.95f, 1.00f, 1.0f);

    /* Labels. */
    char buf[64];
    snprintf(buf, sizeof buf, "t = %.3f", t_cur);
    glColor3f(0.85f, 0.85f, 0.90f);
    gl_text_2d(16, SLIDER_H / 2 + 5, buf);

    glColor3f(0.60f, 0.60f, 0.66f);
    gl_text_2d(x0 - 8, SLIDER_H - 4, "0");
    gl_text_2d(x1 - 8, SLIDER_H - 4, "1");
    gl_text_2d(x0 + (x1 - x0) / 2 - 40, SLIDER_H - 4, "start -------> end");
}

/* LHS sidebar with two vertical rotation-amount sliders. */
static void draw_sidebar(void) {
    int side_y0 = SLIDER_H;
    int side_h  = win_h - SLIDER_H - PLOT_H;

    /* Background. */
    fill_rect_2d(0, side_y0, SIDEBAR_W, side_h, 0.12f, 0.12f, 0.16f, 1.0f);

    /* Right-edge separator. */
    fill_rect_2d(SIDEBAR_W - 1, side_y0, 1, side_h, 0.28f, 0.28f, 0.34f, 1.0f);

    /* Title. */
    glColor3f(0.85f, 0.85f, 0.92f);
    gl_text_2d(8, SLIDER_H + 16, "motion");
    glColor3f(0.60f, 0.60f, 0.66f);
    gl_text_2d(8, SLIDER_H + 30, "angle");

    for (int idx = 0; idx < 2; idx++) {
        int tx  = rot_track_x(idx);
        int top = rot_track_y_top();
        int bot = rot_track_y_bot();
        int ky  = rot_knob_y(idx);

        /* Track (full range, dim). */
        fill_rect_2d(tx - 2, top, 4, bot - top, 0.32f, 0.32f, 0.38f, 1.0f);

        /* Filled portion from knob to bottom. */
        if (idx == 0) glColor4f(0.40f, 0.70f, 1.00f, 1.0f);
        else          glColor4f(1.00f, 0.60f, 0.45f, 1.0f);
        float cr, cg, cb;
        if (idx == 0) { cr = 0.40f; cg = 0.70f; cb = 1.00f; }
        else          { cr = 1.00f; cg = 0.60f; cb = 0.45f; }
        fill_rect_2d(tx - 2, ky, 4, bot - ky, cr, cg, cb, 1.0f);

        /* End-cap ticks. */
        fill_rect_2d(tx - 6, top - 1, 12, 2, 0.65f, 0.65f, 0.72f, 1.0f);
        fill_rect_2d(tx - 6, bot - 1, 12, 2, 0.65f, 0.65f, 0.72f, 1.0f);

        /* Knob. */
        fill_rect_2d(tx - 9, ky - 5, 18, 10, 0.95f, 0.95f, 1.00f, 1.0f);

        /* Label above track. */
        glColor3f(cr, cg, cb);
        gl_text_2d(tx - 4, top - 10, idx == 0 ? "A" : "B");

        /* Value readout below track. */
        char buf[24];
        float v = (idx == 0 ? box_a.motion_angle : box_b.motion_angle);
        snprintf(buf, sizeof buf, "%.2f", v);
        glColor3f(0.85f, 0.85f, 0.92f);
        gl_text_2d(tx - 12, bot + 18, buf);
    }

    /* Bottom hint. */
    glColor3f(0.55f, 0.55f, 0.62f);
    gl_text_2d(4, win_h - PLOT_H - 6, "rad");
}

static void draw_plot(void) {
    /* Background. */
    fill_rect_2d(0, win_h - PLOT_H, win_w, PLOT_H, 0.09f, 0.09f, 0.12f, 1.0f);

    int px0 = plot_x0(), px1 = plot_x1();
    int pyt = plot_y_top(), pyb = plot_y_bot();
    int pw = px1 - px0, ph = pyb - pyt;

    /* Sample separation + Minkowski validity across t. */
    enum { NS = 240 };
    float signed_vals [NS + 1];   /* COM-direction sign correction       */
    float triple_vals [NS + 1];   /* Raw signed scalar triple (no norm)  */
    bool  mink_vals   [NS + 1];

    /* No seed-sign alignment: the raw triple's sign IS the chirality
       of (vB - vA, EA, EB), a continuous geometric property.  Anchoring
       to sign(sep(0)) was brittle under interactive manipulation --
       rotating slightly across the chirality boundary inverted the
       whole curve.  Plotting raw lets you see the actual chirality
       evolve smoothly with rotation; pick a sign convention externally
       (e.g. multiply by sign of an unambiguously-separated frame).   */

    float mn = 1e18f, mx = -1e18f;
    for (int i = 0; i <= NS; i++) {
        float t = (float)i / NS;
        float s  = sep_edge_edge_signed(&box_a, &box_b, t);
        float tr = sep_edge_edge_triple(&box_a, &box_b, t);
        Vec3 na = v_normalize(box_dir_world(&box_a, t, box_a.face_n1_local));
        Vec3 nb = v_normalize(box_dir_world(&box_a, t, box_a.face_n2_local));
        Vec3 nc = v_normalize(box_dir_world(&box_b, t, box_b.face_n1_local));
        Vec3 nd = v_normalize(box_dir_world(&box_b, t, box_b.face_n2_local));
        signed_vals[i]  = s;
        triple_vals[i]  = tr;
        mink_vals[i]    = is_minkowski_face(na, nb, v_neg(nc), v_neg(nd));
        if (s < 999.0f) {
            if (s < mn) mn = s;
            if (s > mx) mx = s;
        }
        if (tr < mn) mn = tr;
        if (tr > mx) mx = tr;
    }
    if (mn > 0.0f) mn = 0.0f;              /* always show zero line */
    if (mx < 0.0f) mx = 0.0f;
    float pad = 0.08f * (mx - mn + 1e-6f);
    mn -= pad; mx += pad;
    float yrange = mx - mn;

    /* Shade plot background by Minkowski-face validity over t.
       Green-tinted where the arcs cross (this edge pair is a valid
       SAT axis), red-tinted where they don't.                      */
    {
        float top_gl = (float)(win_h - pyt);
        float bot_gl = (float)(win_h - pyb);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_QUADS);
        int i = 0;
        while (i <= NS) {
            int j = i;
            while (j < NS && mink_vals[j + 1] == mink_vals[i]) j++;
            float t0 = (float)i / NS;
            float t1 = (float)(j < NS ? j + 1 : NS) / NS;
            float x0 = (float)px0 + (float)pw * t0;
            float x1 = (float)px0 + (float)pw * t1;
            if (mink_vals[i]) glColor4f(0.20f, 0.55f, 0.25f, 0.14f); /* valid  */
            else              glColor4f(0.55f, 0.25f, 0.25f, 0.14f); /* invalid*/
            glVertex2f(x0, bot_gl);
            glVertex2f(x1, bot_gl);
            glVertex2f(x1, top_gl);
            glVertex2f(x0, top_gl);
            i = j + 1;
        }
        glEnd();
        glDisable(GL_BLEND);
    }

    /* Frame. */
    glColor3f(0.35f, 0.35f, 0.42f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f((float)px0, (float)(win_h - pyt));
    glVertex2f((float)px1, (float)(win_h - pyt));
    glVertex2f((float)px1, (float)(win_h - pyb));
    glVertex2f((float)px0, (float)(win_h - pyb));
    glEnd();

    /* Zero line (separation = 0, i.e. touching). */
    float zero_y = (float)pyb - (float)ph * (0.0f - mn) / yrange;
    glColor3f(0.55f, 0.35f, 0.35f);
    glLineWidth(1.2f);
    glBegin(GL_LINES);
    glVertex2f((float)px0, (float)win_h - zero_y);
    glVertex2f((float)px1, (float)win_h - zero_y);
    glEnd();

    /* Gridlines: t = 0.25, 0.5, 0.75. */
    glColor3f(0.25f, 0.25f, 0.30f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int k = 1; k < 4; k++) {
        float x = (float)px0 + (float)pw * (k / 4.0f);
        glVertex2f(x, (float)win_h - pyt);
        glVertex2f(x, (float)win_h - pyb);
    }
    glEnd();

    /* COM-direction sign correction: cyan.  Has spurious flips
       when cross(EA, EB) sweeps perpendicular to (comB - comA). */
    glColor3f(0.30f, 0.85f, 1.00f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= NS; i++) {
        float t = (float)i / NS;
        float v = signed_vals[i];
        if (v > 999.0f) continue;
        float x = (float)px0 + (float)pw * t;
        float y = (float)pyb - (float)ph * (v - mn) / yrange;
        glVertex2f(x, (float)win_h - y);
    }
    glEnd();

    /* Raw signed scalar triple product (no normalisation), aligned
       with sign at t=0 so positive = separated.  Smooth polynomial
       in motion params: free of the rotational parity sign flips
       that produce false roots in CCD.                             */
    glColor3f(1.00f, 0.65f, 0.20f);
    glLineWidth(2.5f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= NS; i++) {
        float t = (float)i / NS;
        float v = triple_vals[i];
        float x = (float)px0 + (float)pw * t;
        float y = (float)pyb - (float)ph * (v - mn) / yrange;
        glVertex2f(x, (float)win_h - y);
    }
    glEnd();

    /* Current-time vertical marker. */
    float tx = (float)px0 + (float)pw * t_cur;
    glColor3f(0.95f, 0.95f, 1.00f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    glVertex2f(tx, (float)win_h - pyt);
    glVertex2f(tx, (float)win_h - pyb);
    glEnd();

    /* Current-time readout dot on the orange (triple) curve. */
    float s_now = sep_edge_edge_triple(&box_a, &box_b, t_cur);
    {
        float y = (float)pyb - (float)ph * (s_now - mn) / yrange;
        glPointSize(8.0f);
        glColor3f(1.0f, 1.0f, 0.3f);
        glBegin(GL_POINTS);
        glVertex2f(tx, (float)win_h - y);
        glEnd();
    }

    /* Labels. */
    char buf[96];
    glColor3f(0.80f, 0.80f, 0.85f);
    gl_text_2d(6, pyt - 4, "sep");

    snprintf(buf, sizeof buf, "max %+.3f", mx);
    gl_text_2d(6, pyt + 12, buf);
    snprintf(buf, sizeof buf, "min %+.3f", mn);
    gl_text_2d(6, pyb - 2, buf);

    glColor3f(0.60f, 0.60f, 0.66f);
    gl_text_2d(px0 - 2, pyb + 16, "t=0");
    gl_text_2d(px1 - 30, pyb + 16, "t=1");

    /* Legend. */
    glColor3f(0.30f, 0.85f, 1.00f);
    gl_text_2d(px1 - 300, pyt + 14, "COM-direction sign correction");
    glColor3f(1.00f, 0.65f, 0.20f);
    gl_text_2d(px1 - 300, pyt + 30, "signed triple (no normalisation)");
    glColor3f(0.45f, 0.80f, 0.45f);
    gl_text_2d(px1 - 300, pyt + 46, "green band: arcs intersect");
    glColor3f(0.85f, 0.50f, 0.50f);
    gl_text_2d(px1 - 300, pyt + 62, "red band: arcs don't");

    /* Current value readout (top-left of plot). */
    if (s_now < 999.0f) {
        snprintf(buf, sizeof buf, "sep(t) = %+.4f", s_now);
    } else {
        snprintf(buf, sizeof buf, "sep(t) = parallel");
    }
    if (s_now < 0.0f && s_now < 999.0f) glColor3f(1.00f, 0.35f, 0.35f);
    else                                 glColor3f(1.00f, 1.00f, 0.35f);
    gl_text_2d(px0 + 8, pyt + 14, buf);
}

/* =================================================================
 *  Gauss-map overlay
 * =================================================================
 * A small inset in the upper-right corner of the 3-D view showing the
 * unit sphere with the two great-circle arcs formed by the face
 * normals of each edge.  Arc A runs a -> b, where a,b are A's face
 * normals at time t.  Arc B runs -c -> -d (Minkowski-negated) so the
 * two arcs cross iff the Minkowski face is valid (and the crossing
 * point is the separation-axis direction).
 */

static void draw_sphere_wire_inset(void) {
    enum { NLAT = 10, NLON = 12, SEG = 48 };
    int i, j;

    glColor4f(0.35f, 0.35f, 0.42f, 0.55f);
    glLineWidth(1.0f);

    for (i = 1; i < NLAT; i++) {
        float phi = (float)M_PI * i / NLAT;
        float y   = cosf(phi);
        float rr  = sinf(phi);
        glBegin(GL_LINE_LOOP);
        for (j = 0; j < SEG; j++) {
            float th = 2.0f * (float)M_PI * j / SEG;
            glVertex3f(rr * cosf(th), y, rr * sinf(th));
        }
        glEnd();
    }
    for (j = 0; j < NLON; j++) {
        float th = 2.0f * (float)M_PI * j / NLON;
        glBegin(GL_LINE_STRIP);
        for (i = 0; i <= SEG; i++) {
            float phi = (float)M_PI * i / SEG;
            glVertex3f(sinf(phi) * cosf(th),
                       cosf(phi),
                       sinf(phi) * sinf(th));
        }
        glEnd();
    }
}

static void draw_sphere_solid_inset(void) {
    static GLUquadric *q = NULL;
    if (!q) q = gluNewQuadric();
    gluQuadricDrawStyle(q, GLU_FILL);
    gluQuadricNormals(q, GLU_SMOOTH);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    {   float pos[]  = { 2, 3, 4, 0 };
        float amb[]  = { 0.05f, 0.05f, 0.07f, 1 };
        float diff[] = { 0.18f, 0.18f, 0.21f, 1 };
        float spec[] = { 0.10f, 0.10f, 0.10f, 1 };
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glLightfv(GL_LIGHT0, GL_AMBIENT,  amb);
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  diff);
        glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
    }
    {   float d[] = { 0.18f, 0.18f, 0.22f, 1 };
        float s[] = { 0.30f, 0.30f, 0.30f, 1 };
        float a[] = { 0.06f, 0.06f, 0.08f, 1 };
        glMaterialfv(GL_FRONT, GL_DIFFUSE,  d);
        glMaterialfv(GL_FRONT, GL_SPECULAR, s);
        glMaterialfv(GL_FRONT, GL_AMBIENT,  a);
        glMaterialf (GL_FRONT, GL_SHININESS, 32.0f);
    }
    gluSphere(q, 0.992, 40, 20);
    glDisable(GL_LIGHTING);
}

static void draw_arc_inset(Vec3 a, Vec3 b,
                           float cr, float cg, float cb, float lw) {
    enum { N = 64 };
    float rr = 1.004f;
    glColor3f(cr, cg, cb);
    glLineWidth(lw);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= N; i++) {
        Vec3 p = v_scale(v_normalize(v_slerp(a, b, (float)i / N)), rr);
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
}

static void draw_point_inset(Vec3 p, float cr, float cg, float cb, float sz) {
    Vec3 pp = v_scale(v_normalize(p), 1.015f);
    glPointSize(sz);
    glColor3f(cr, cg, cb);
    glBegin(GL_POINTS);
    glVertex3f(pp.x, pp.y, pp.z);
    glEnd();
}

/* Inset bounds in window (top-left origin) pixels, including the title
   strip.  Used for drawing AND hit-testing so the two can't drift.    */
static void inset_rect(int *x, int *y, int *w, int *h) {
    *x = win_w - INSET_W - INSET_PAD;
    *y = SLIDER_H + 14;
    *w = INSET_W;
    *h = INSET_H + INSET_TITLE;
}

static bool inset_hit(int mx, int my) {
    int ix, iy, iw, ih;
    inset_rect(&ix, &iy, &iw, &ih);
    return mx >= ix - 6 && mx <= ix + iw + 6 &&
           my >= iy - 6 && my <= iy + ih + 6;
}

static void draw_gauss_map_overlay(void) {
    int iw  = INSET_W;
    int ih  = INSET_H;
    int pad = INSET_PAD;
    int px  = win_w - iw - pad;                      /* top-left X */
    int py  = SLIDER_H + 14;                         /* top-left Y */
    int title_h = INSET_TITLE;

    /* Panel background + border in 2-D ortho. */
    setup_ortho_full();
    glDisable(GL_DEPTH_TEST);

    fill_rect_2d(px - 6, py - 6, iw + 12, ih + title_h + 12,
                 0.05f, 0.05f, 0.08f, 0.92f);

    glLineWidth(1.0f);
    glColor3f(0.35f, 0.35f, 0.42f);
    glBegin(GL_LINE_LOOP);
    glVertex2f((float)(px - 6),          (float)(win_h - (py - 6)));
    glVertex2f((float)(px + iw + 6),     (float)(win_h - (py - 6)));
    glVertex2f((float)(px + iw + 6),     (float)(win_h - (py + ih + title_h + 6)));
    glVertex2f((float)(px - 6),          (float)(win_h - (py + ih + title_h + 6)));
    glEnd();

    glColor3f(0.85f, 0.85f, 0.92f);
    gl_text_2d(px + 4, py + 14, "Gauss Map");

    /* Compute world-space normals / edge directions at t_cur. */
    Vec3 a = v_normalize(box_dir_world(&box_a, t_cur, box_a.face_n1_local));
    Vec3 b = v_normalize(box_dir_world(&box_a, t_cur, box_a.face_n2_local));
    Vec3 c = v_normalize(box_dir_world(&box_b, t_cur, box_b.face_n1_local));
    Vec3 d = v_normalize(box_dir_world(&box_b, t_cur, box_b.face_n2_local));
    Vec3 nc = v_neg(c), nd = v_neg(d);

    /* Sign-corrected separation axis: direction of cross(dA, dB)
       oriented to point from comA to comB.                        */
    Vec3 pA0, pA1, pB0, pB1;
    box_edge_world(&box_a, t_cur, &pA0, &pA1);
    box_edge_world(&box_b, t_cur, &pB0, &pB1);
    Vec3 dA = v_normalize(v_sub(pA1, pA0));
    Vec3 dB = v_normalize(v_sub(pB1, pB0));
    Vec3 axis = v_cross(dA, dB);
    float axis_len = v_len(axis);
    Vec3 axis_hat = axis_len > 1e-4f ? v_scale(axis, 1.0f/axis_len) : v3(0,0,0);
    Vec3 comA, comB; Quat rA, rB;
    box_pose_at(&box_a, t_cur, &comA, &rA);
    box_pose_at(&box_b, t_cur, &comB, &rB);
    if (axis_len > 1e-4f && v_dot(axis_hat, v_sub(comB, comA)) < 0.0f)
        axis_hat = v_neg(axis_hat);

    /* Pass -c, -d so the test reads "arc a->b crosses arc -c->-d" --
       the actual Minkowski-difference face condition.                 */
    bool mink = is_minkowski_face(a, b, nc, nd);

    /* Sub-viewport in GL (bottom-left origin). */
    int gl_vx = px;
    int gl_vy = win_h - (py + title_h + ih);
    int gl_vw = iw;
    int gl_vh = ih;

    /* Scissor keeps the depth clear + draws confined to the inset. */
    glEnable(GL_SCISSOR_TEST);
    glScissor(gl_vx, gl_vy, gl_vw, gl_vh);
    glViewport(gl_vx, gl_vy, gl_vw, gl_vh);

    glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(38.0, 1.0, 0.1, 20.0);

    /* Independent orbit -- drag inside the inset to spin it on its
       own, leaving the main 3-D camera alone.                        */
    float cx = cam_dist_g * cosf(cam_el_g) * sinf(cam_az_g);
    float cy = cam_dist_g * sinf(cam_el_g);
    float cz = cam_dist_g * cosf(cam_el_g) * cosf(cam_az_g);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    draw_sphere_solid_inset();
    draw_sphere_wire_inset();

    /* Arcs.  Colour green when the Minkowski face is valid. */
    if (mink) {
        draw_arc_inset(a,  b,  0.15f, 0.95f, 0.35f, 3.0f);
        draw_arc_inset(nc, nd, 0.15f, 0.95f, 0.35f, 3.0f);
    } else {
        draw_arc_inset(a,  b,  0.35f, 0.70f, 1.00f, 3.0f);   /* A : blue */
        draw_arc_inset(nc, nd, 1.00f, 0.40f, 0.45f, 3.0f);   /* B : red  */
    }

    /* Endpoints. */
    draw_point_inset(a,  0.40f, 0.75f, 1.00f, 9.0f);
    draw_point_inset(b,  0.20f, 0.40f, 0.95f, 9.0f);
    draw_point_inset(nc, 1.00f, 0.55f, 0.50f, 9.0f);
    draw_point_inset(nd, 0.95f, 0.25f, 0.30f, 9.0f);

    /* Separation-axis direction (sign-corrected) as a yellow dot. */
    if (axis_len > 1e-4f) {
        draw_point_inset(axis_hat, 0.95f, 0.95f, 0.20f, 11.0f);
        /* And its antipode faintly, since cross can pick either pole. */
        draw_point_inset(v_neg(axis_hat), 0.45f, 0.45f, 0.10f, 6.0f);
    }

    /* Labels drawn without depth test so they always show. */
    glDisable(GL_DEPTH_TEST);
    Vec3 la  = v_scale(v_normalize(a),  1.15f);
    Vec3 lb  = v_scale(v_normalize(b),  1.15f);
    Vec3 lnc = v_scale(v_normalize(nc), 1.15f);
    Vec3 lnd = v_scale(v_normalize(nd), 1.15f);
    glColor3f(0.40f, 0.75f, 1.00f); gl_text_3d(la,  "a");
    glColor3f(0.20f, 0.40f, 0.95f); gl_text_3d(lb,  "b");
    glColor3f(1.00f, 0.55f, 0.50f); gl_text_3d(lnc, "-c");
    glColor3f(0.95f, 0.25f, 0.30f); gl_text_3d(lnd, "-d");
    if (axis_len > 1e-4f) {
        Vec3 lax = v_scale(axis_hat, 1.18f);
        glColor3f(1.00f, 1.00f, 0.35f);
        gl_text_3d(lax, "n");
    }

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);

    /* Minkowski status line under the inset title. */
    setup_ortho_full();
    if (mink) {
        glColor3f(0.15f, 0.95f, 0.35f);
        gl_text_2d(px + 100, py + 14, "valid face");
    } else {
        glColor3f(0.80f, 0.45f, 0.45f);
        gl_text_2d(px + 100, py + 14, "no face");
    }
}

/* =================================================================
 *  Render
 * ================================================================= */

/* Full box-box SAT following Dirk Gregorius' lmHulltoHull logic
   (GDC 2013).  Checks 3 face axes of A, 3 of B, plus 9 edge-edge
   cross axes.  Boxes overlap iff the max separation across ALL 15
   axes is <= 0.  Edge-edge classification follows Dirk's dispatch:
   edge query separation beats both face queries by k_tol.           */

typedef struct {
    bool  colliding;        /* all axes have sep <= 0                  */
    bool  edge_is_witness;  /* colliding AND edge axis is deepest      */
    float best_face_sep;    /* max separation across all face axes     */
    float best_edge_sep;    /* max separation across all edge axes     */
} SatInfo;

/* Box-box SAT gap along a (need not be unit) axis.  Signed:
     > 0  intervals disjoint by that amount
     <= 0 intervals overlap.
   Returns -1e30 for a degenerate (zero-length) axis.                 */
static float sat_axis_sep(Vec3 axA[3], Vec3 axB[3],
                           const float hA[3], const float hB[3],
                           Vec3 d, Vec3 axis_raw) {
    float len = v_len(axis_raw);
    if (len < 1e-6f) return -1e30f;
    Vec3 n = v_scale(axis_raw, 1.0f / len);

    float hAproj = hA[0] * fabsf(v_dot(axA[0], n))
                 + hA[1] * fabsf(v_dot(axA[1], n))
                 + hA[2] * fabsf(v_dot(axA[2], n));
    float hBproj = hB[0] * fabsf(v_dot(axB[0], n))
                 + hB[1] * fabsf(v_dot(axB[1], n))
                 + hB[2] * fabsf(v_dot(axB[2], n));
    return fabsf(v_dot(d, n)) - hAproj - hBproj;
}

static SatInfo sat_query(float t) {
    SatInfo r = { false, false, -1e30f, -1e30f };

    Vec3 comA, comB; Quat rA, rB;
    box_pose_at(&box_a, t, &comA, &rA);
    box_pose_at(&box_b, t, &comB, &rB);

    Vec3 axA[3] = {
        q_rotate(rA, v3(1.0f, 0.0f, 0.0f)),
        q_rotate(rA, v3(0.0f, 1.0f, 0.0f)),
        q_rotate(rA, v3(0.0f, 0.0f, 1.0f))
    };
    Vec3 axB[3] = {
        q_rotate(rB, v3(1.0f, 0.0f, 0.0f)),
        q_rotate(rB, v3(0.0f, 1.0f, 0.0f)),
        q_rotate(rB, v3(0.0f, 0.0f, 1.0f))
    };
    float hA[3] = { box_a.size.x, box_a.size.y, box_a.size.z };
    float hB[3] = { box_b.size.x, box_b.size.y, box_b.size.z };
    Vec3 d = v_sub(comB, comA);

    /* Face axes (3 of A, 3 of B). */
    for (int i = 0; i < 3; i++) {
        float s = sat_axis_sep(axA, axB, hA, hB, d, axA[i]);
        if (s > r.best_face_sep) r.best_face_sep = s;
        s = sat_axis_sep(axA, axB, hA, hB, d, axB[i]);
        if (s > r.best_face_sep) r.best_face_sep = s;
    }

    /* 9 edge-edge cross axes. */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Vec3 cr = v_cross(axA[i], axB[j]);
            float s = sat_axis_sep(axA, axB, hA, hB, d, cr);
            if (s > r.best_edge_sep) r.best_edge_sep = s;
        }
    }

    r.colliding = (r.best_face_sep <= 0.0f) && (r.best_edge_sep <= 0.0f);
    /* Dirk's edge-vs-face dispatch tolerance. */
    const float K_TOL = 0.05f;
    r.edge_is_witness = r.colliding &&
                        (r.best_edge_sep > r.best_face_sep + K_TOL);
    return r;
}

static void render(void) {
    SatInfo sat = sat_query(t_cur);
    bool colliding = sat.colliding;

    /* Red tint on the background whenever full SAT reports overlap. */
    if (colliding) glClearColor(0.17f, 0.08f, 0.09f, 1.0f);
    else           glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* ---------------- 3-D scene ---------------- */
    setup_camera_3d();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

    draw_ground();

    /* Faint start/end ghosts so motion is obvious. */
    draw_ghost_box(&box_a, 0.0f, 0.40f, 0.65f, 1.00f);
    draw_ghost_box(&box_a, 1.0f, 0.40f, 0.65f, 1.00f);
    draw_ghost_box(&box_b, 0.0f, 1.00f, 0.55f, 0.45f);
    draw_ghost_box(&box_b, 1.0f, 1.00f, 0.55f, 0.45f);

    /* Current pose.  Highlight edges hot-red when colliding. */
    if (colliding) {
        draw_box(&box_a, t_cur,
                 0.40f, 0.65f, 1.00f,
                 1.00f, 0.25f, 0.25f);
        draw_box(&box_b, t_cur,
                 1.00f, 0.55f, 0.45f,
                 1.00f, 0.25f, 0.25f);
    } else {
        draw_box(&box_a, t_cur,
                 0.40f, 0.65f, 1.00f,
                 0.30f, 0.90f, 1.00f);
        draw_box(&box_b, t_cur,
                 1.00f, 0.55f, 0.45f,
                 1.00f, 0.80f, 0.30f);
    }

    /* Edge-line visualisation:
       - Faint dashed infinite extension of each edge in box colour,
         so you can see the lines the SAT formula reasons about.
       - Bright yellow segment between the two closest points on
         those lines -- this IS the common perpendicular, so its
         length equals the line-line distance and its direction is
         the (sign-corrected) cross-axis n_hat.                      */
    {
        Vec3 pA0, pA1, pB0, pB1;
        box_edge_world(&box_a, t_cur, &pA0, &pA1);
        box_edge_world(&box_b, t_cur, &pB0, &pB1);
        Vec3 pA = v_scale(v_add(pA0, pA1), 0.5f);
        Vec3 pB = v_scale(v_add(pB0, pB1), 0.5f);
        Vec3 dA = v_normalize(v_sub(pA1, pA0));
        Vec3 dB = v_normalize(v_sub(pB1, pB0));

        const float EXT = 12.0f;
        Vec3 aFar0 = v_add(pA, v_scale(dA, -EXT));
        Vec3 aFar1 = v_add(pA, v_scale(dA,  EXT));
        Vec3 bFar0 = v_add(pB, v_scale(dB, -EXT));
        Vec3 bFar1 = v_add(pB, v_scale(dB,  EXT));

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
#ifndef __EMSCRIPTEN__
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0x00FF);
#endif
        glLineWidth(1.0f);

        /* Box A's edge line (faint blue). */
        glColor4f(0.40f, 0.65f, 1.00f, 0.40f);
        glBegin(GL_LINES);
        glVertex3f(aFar0.x, aFar0.y, aFar0.z);
        glVertex3f(aFar1.x, aFar1.y, aFar1.z);
        glEnd();

        /* Box B's edge line (faint orange). */
        glColor4f(1.00f, 0.55f, 0.45f, 0.40f);
        glBegin(GL_LINES);
        glVertex3f(bFar0.x, bFar0.y, bFar0.z);
        glVertex3f(bFar1.x, bFar1.y, bFar1.z);
        glEnd();

#ifndef __EMSCRIPTEN__
        glDisable(GL_LINE_STIPPLE);
#endif
        glDisable(GL_BLEND);

        /* Yellow common-perpendicular segment between the two
           closest points on the (infinite) edge lines.             */
        float s_par, t_par;
        if (line_closest_params(pA, dA, pB, dB, &s_par, &t_par)) {
            Vec3 cA = v_add(pA, v_scale(dA, s_par));
            Vec3 cB = v_add(pB, v_scale(dB, t_par));

            glLineWidth(2.2f);
            glColor3f(0.95f, 0.95f, 0.35f);
            glBegin(GL_LINES);
            glVertex3f(cA.x, cA.y, cA.z);
            glVertex3f(cB.x, cB.y, cB.z);
            glEnd();

            glPointSize(7.0f);
            glBegin(GL_POINTS);
            glVertex3f(cA.x, cA.y, cA.z);
            glVertex3f(cB.x, cB.y, cB.z);
            glEnd();
        }
    }

    /* Box labels (disable depth so they always show). */
    glDisable(GL_DEPTH_TEST);
    {
        Vec3 ca, cb; Quat ra, rb;
        box_pose_at(&box_a, t_cur, &ca, &ra);
        box_pose_at(&box_b, t_cur, &cb, &rb);
        ca.y += box_a.size.y + 0.25f;
        cb.y += box_b.size.y + 0.25f;
        glColor3f(0.60f, 0.85f, 1.00f);
        gl_text_3d(ca, "A");
        glColor3f(1.00f, 0.75f, 0.55f);
        gl_text_3d(cb, "B");
    }
    glEnable(GL_DEPTH_TEST);

    /* ---------------- Gauss-map inset ---------------- */
    draw_gauss_map_overlay();

    /* ---------------- 2-D overlays ---------------- */
    glViewport(0, 0, win_w, win_h);
    glDisable(GL_DEPTH_TEST);
    setup_ortho_full();

    draw_slider();
    draw_sidebar();
    draw_plot();

    /* HUD strip between 3-D view and slider. */
    glColor3f(0.60f, 0.60f, 0.66f);
    gl_text_2d(10, SLIDER_H + 18,
        "LMB orbit   RMB trackball-rotate box   Wheel zoom   R reset   Esc quit");

    if (colliding) {
        glColor3f(1.00f, 0.35f, 0.35f);
        gl_text_2d(win_w / 2 - 70, SLIDER_H + 18,
                   sat.edge_is_witness
                       ? "COLLIDING (edge-edge)"
                       : "COLLIDING (face contact)");
    }

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POINT_SMOOTH);

    plat_swap();
}

/* =================================================================
 *  Interaction: virtual-trackball rotation
 * =================================================================
 * Right-drag rotates a box using Gems-style virtual-trackball logic
 * centred at the box's current-time centre.  The same rotation delta
 * is applied to rot_start and rot_end so the animation motion is
 * preserved.                                                        */

static void trackball_rotate_box(Box *b,
                                  int mx_old, int my_old,
                                  int mx_new, int my_new) {
    Vec3 cop = camera_position_world();
    Vec3 cor; Quat tmp;
    box_pose_at(b, t_cur, &cor, &tmp);

    Vec3 ro_a, rd_a, ro_b, rd_b;
    screen_to_ray(mx_old, my_old, &ro_a, &rd_a);
    screen_to_ray(mx_new, my_new, &ro_b, &rd_b);

    Quat delta = virtual_trackball(cop, cor, rd_a, rd_b);
    b->rot_start = q_normalize(q_mul(delta, b->rot_start));
    b->rot_end   = q_normalize(q_mul(delta, b->rot_end));
}

/* =================================================================
 *  Shared event handlers (called from both Win32 wndproc and GLFW).
 * ================================================================= */

static void on_resize(int w, int h) {
    win_w = w;
    win_h = (h < 1) ? 1 : h;
    glViewport(0, 0, win_w, win_h);
    PLAT_INVALIDATE();
}

static void on_mouse_down(int button, int mx, int my) {
    mx_prev = mx; my_prev = my;
    if (button == 0) {
        if (slider_hit(mx, my) || my < SLIDER_H) {
            slider_drag = true;
            slider_update(mx);
        } else if (plot_hit(my)) {
            slider_drag = true;
            int px0 = plot_x0(), px1 = plot_x1();
            float tv = (float)(mx - px0) / (float)(px1 - px0);
            if (tv < 0) tv = 0;
            if (tv > 1) tv = 1;
            t_cur = tv;
        } else if (rot_slider_hit(mx, my, 0)) {
            rot_slider_drag = 0;
            rot_slider_update(my, 0);
        } else if (rot_slider_hit(mx, my, 1)) {
            rot_slider_drag = 1;
            rot_slider_update(my, 1);
        } else if (inset_hit(mx, my)) {
            inset_drag = true;
        } else {
            lmb_down = true;
        }
    } else if (button == 1) {
        if (my > SLIDER_H && !plot_hit(my) && !inset_hit(mx, my)
                          && mx > SIDEBAR_W) {
            rmb_down = true;
            box_drag_idx = pick_box(mx, my);
        }
    }
    PLAT_INVALIDATE();
}

static void on_mouse_up(int button) {
    if (button == 0) {
        lmb_down = false;
        slider_drag = false;
        inset_drag = false;
        rot_slider_drag = -1;
    } else if (button == 1) {
        rmb_down = false;
        box_drag_idx = -1;
    }
}

static void on_mouse_move(int mx, int my) {
    int dx = mx - mx_prev;
    int dy = my - my_prev;

    if (slider_drag) {
        if (my < SLIDER_H + 40) {
            slider_update(mx);
        } else if (plot_hit(my)) {
            int px0 = plot_x0(), px1 = plot_x1();
            float tv = (float)(mx - px0) / (float)(px1 - px0);
            if (tv < 0) tv = 0;
            if (tv > 1) tv = 1;
            t_cur = tv;
        } else {
            slider_update(mx);
        }
        PLAT_INVALIDATE();
    } else if (rot_slider_drag >= 0) {
        rot_slider_update(my, rot_slider_drag);
        PLAT_INVALIDATE();
    } else if (inset_drag) {
        cam_az_g -= dx * 0.008f;
        cam_el_g += dy * 0.008f;
        if (cam_el_g >  1.5f) cam_el_g =  1.5f;
        if (cam_el_g < -1.5f) cam_el_g = -1.5f;
        PLAT_INVALIDATE();
    } else if (lmb_down) {
        cam_az -= dx * 0.005f;
        cam_el += dy * 0.005f;
        if (cam_el >  1.5f) cam_el =  1.5f;
        if (cam_el < -1.5f) cam_el = -1.5f;
        PLAT_INVALIDATE();
    } else if (rmb_down && box_drag_idx >= 0) {
        Box *b = (box_drag_idx == 0) ? &box_a : &box_b;
        trackball_rotate_box(b, mx_prev, my_prev, mx, my);
        PLAT_INVALIDATE();
    }

    mx_prev = mx;
    my_prev = my;
}

/* `delta` is normalised: 1.0 = one wheel click up, -1.0 = one click down. */
static void on_mouse_wheel(float delta, int mx, int my) {
    if (inset_hit(mx, my)) {
        cam_dist_g -= delta * 0.24f;
        if (cam_dist_g < 1.8f) cam_dist_g = 1.8f;
        if (cam_dist_g > 8.0f) cam_dist_g = 8.0f;
    } else {
        cam_dist -= delta * 0.60f;
        if (cam_dist < 2.0f)  cam_dist = 2.0f;
        if (cam_dist > 20.0f) cam_dist = 20.0f;
    }
    PLAT_INVALIDATE();
}

static void on_key_down(int key) {
    if (key == KEY_ESCAPE) {
#ifndef __EMSCRIPTEN__
        PostQuitMessage(0);
#endif
    } else if (key == KEY_R) {
        reset_scene();
        PLAT_INVALIDATE();
    } else if (key == KEY_LEFT) {
        t_cur -= 0.01f;
        if (t_cur < 0) t_cur = 0;
        PLAT_INVALIDATE();
    } else if (key == KEY_RIGHT) {
        t_cur += 0.01f;
        if (t_cur > 1) t_cur = 1;
        PLAT_INVALIDATE();
    }
}

/* =================================================================
 *  Platform layer: windowing, main loop, entry point
 * ================================================================= */

#ifdef __EMSCRIPTEN__

static void web_fb_size(GLFWwindow *w, int ww, int hh) {
    (void)w; on_resize(ww, hh);
}

static void web_cursor_pos(GLFWwindow *w, double x, double y) {
    (void)w; on_mouse_move((int)x, (int)y);
}

static void web_mouse_button(GLFWwindow *w, int b, int action, int mods) {
    (void)w; (void)mods;
    double x, y;
    glfwGetCursorPos(w, &x, &y);
    int btn = (b == GLFW_MOUSE_BUTTON_LEFT) ? 0
            : (b == GLFW_MOUSE_BUTTON_RIGHT) ? 1 : -1;
    if (btn < 0) return;
    if (action == GLFW_PRESS)        on_mouse_down(btn, (int)x, (int)y);
    else if (action == GLFW_RELEASE) on_mouse_up(btn);
}

static void web_scroll(GLFWwindow *w, double dx, double dy) {
    (void)dx;
    double x, y;
    glfwGetCursorPos(w, &x, &y);
    on_mouse_wheel((float)dy, (int)x, (int)y);
}

static void web_key(GLFWwindow *w, int key, int sc, int action, int mods) {
    (void)w; (void)sc; (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    int k = 0;
    switch (key) {
        case GLFW_KEY_ESCAPE: k = KEY_ESCAPE; break;
        case GLFW_KEY_R:      k = KEY_R;      break;
        case GLFW_KEY_LEFT:   k = KEY_LEFT;   break;
        case GLFW_KEY_RIGHT:  k = KEY_RIGHT;  break;
        default: return;
    }
    on_key_down(k);
}

static void web_frame(void) {
    glfwPollEvents();
    render();
}

int main(void) {
    reset_scene();

    if (!glfwInit()) return 1;
    g_window = glfwCreateWindow(win_w, win_h,
        "Edge-Edge Motion  |  Signed Separation", NULL, NULL);
    if (!g_window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(g_window);

    glfwSetFramebufferSizeCallback(g_window, web_fb_size);
    glfwSetCursorPosCallback      (g_window, web_cursor_pos);
    glfwSetMouseButtonCallback    (g_window, web_mouse_button);
    glfwSetScrollCallback         (g_window, web_scroll);
    glfwSetKeyCallback            (g_window, web_key);

    init_font();
    emscripten_set_main_loop(web_frame, 0, 1);
    return 0;
}

#else  /* Windows */

static LRESULT CALLBACK wndproc(HWND hw, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        on_resize(LOWORD(lp), HIWORD(lp));
        return 0;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(hw, &ps);
        render();
        EndPaint(hw, &ps);
        return 0;
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_LBUTTONDOWN:
        on_mouse_down(0, (short)LOWORD(lp), (short)HIWORD(lp));
        SetCapture(hw);
        return 0;

    case WM_LBUTTONUP:
        on_mouse_up(0);
        ReleaseCapture();
        return 0;

    case WM_RBUTTONDOWN:
        on_mouse_down(1, (short)LOWORD(lp), (short)HIWORD(lp));
        if (rmb_down) SetCapture(hw);
        return 0;

    case WM_RBUTTONUP:
        on_mouse_up(1);
        ReleaseCapture();
        return 0;

    case WM_MOUSEMOVE:
        on_mouse_move((short)LOWORD(lp), (short)HIWORD(lp));
        return 0;

    case WM_MOUSEWHEEL: {
        short delta = (short)HIWORD(wp);
        POINT pt = { (short)LOWORD(lp), (short)HIWORD(lp) };
        ScreenToClient(hw, &pt);
        on_mouse_wheel((float)delta / (float)WHEEL_DELTA, pt.x, pt.y);
        return 0;
    }

    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) on_key_down(KEY_ESCAPE);
        else if (wp == 'R')  on_key_down(KEY_R);
        else if (wp == VK_LEFT)  on_key_down(KEY_LEFT);
        else if (wp == VK_RIGHT) on_key_down(KEY_RIGHT);
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hw, msg, wp, lp);
}

static bool init_gl(HWND hw) {
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    g_hdc = GetDC(hw);
    int pf = ChoosePixelFormat(g_hdc, &pfd);
    if (!pf) return false;
    SetPixelFormat(g_hdc, pf, &pfd);

    g_hrc = wglCreateContext(g_hdc);
    if (!g_hrc) return false;
    wglMakeCurrent(g_hdc, g_hrc);
    return true;
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmd, int show) {
    (void)prev; (void)cmd;

    reset_scene();

    WNDCLASSA wc    = {0};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = wndproc;
    wc.hInstance     = inst;
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "GaussMapViz";
    RegisterClassA(&wc);

    g_hwnd = CreateWindowA(
        "GaussMapViz", "Edge-Edge Motion  |  Signed Separation",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, win_w, win_h,
        NULL, NULL, inst, NULL);

    if (!init_gl(g_hwnd)) {
        MessageBoxA(NULL, "Failed to create OpenGL context.", "Error", MB_OK);
        return 1;
    }

    init_font();
    ShowWindow(g_hwnd, show);
    UpdateWindow(g_hwnd);

    MSG m;
    while (GetMessageA(&m, NULL, 0, 0)) {
        TranslateMessage(&m);
        DispatchMessageA(&m);
    }

    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(g_hrc);
    ReleaseDC(g_hwnd, g_hdc);
    return (int)m.wParam;
}

#endif
