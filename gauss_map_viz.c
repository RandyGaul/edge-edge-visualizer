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
 * Build
 *   Desktop (Win32/macOS/Linux):  cmake ... && cmake --build .
 *   Web (Emscripten via emsdk):   web.cmd    (or emcmake cmake ...)
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
  #include <SDL.h>
  #include <GLES3/gl3.h>
#else
  #include <SDL.h>
  #include <SDL_opengl.h>   /* brings in basic types + GL 1.1 symbols */
#endif

#include "vendor/stb_easy_font.h"

/* GLdouble isn't in GLES3 / GL 3.3 core; define for legacy code paths. */
#ifndef GLdouble
  typedef double GLdouble;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int win_w = 1200, win_h = 820;

/* =================================================================
 *  Legacy fixed-function enum shims.  ES3 / GL 3.3 core headers do
 *  not define these, but we still use them as sentinel values in
 *  the viz_* wrappers below (and in the app code that calls them).
 * ================================================================= */

#ifndef GL_QUADS
  #define GL_QUADS              0x0007
  #define GL_QUAD_STRIP         0x0008
  #define GL_POLYGON            0x0009
#endif
#ifndef GL_PROJECTION
  #define GL_MODELVIEW          0x1700
  #define GL_PROJECTION         0x1701
#endif
#ifndef GL_MODELVIEW_MATRIX
  #define GL_MODELVIEW_MATRIX   0x0BA6
  #define GL_PROJECTION_MATRIX  0x0BA7
#endif
#ifndef GL_LIGHTING
  #define GL_LIGHTING           0x0B50
  #define GL_LIGHT0             0x4000
#endif
#ifndef GL_LINE_SMOOTH
  #define GL_LINE_SMOOTH        0x0B20
  #define GL_POINT_SMOOTH       0x0B10
  #define GL_LINE_STIPPLE       0x0B24
  #define GL_LINE_SMOOTH_HINT   0x0C52
#endif
#ifndef GL_POSITION
  #define GL_POSITION           0x1203
  #define GL_AMBIENT            0x1200
  #define GL_DIFFUSE            0x1201
  #define GL_SPECULAR           0x1202
  #define GL_SHININESS          0x1601
  #define GL_FRONT              0x0404
#endif

/* =================================================================
 *  GL 3.3+ function pointer loader (desktop only).
 *  On web, <GLES3/gl3.h> provides these directly.
 * ================================================================= */

#ifndef __EMSCRIPTEN__

typedef char     GLchar_;
typedef ptrdiff_t GLsizeiptr_;
typedef ptrdiff_t GLintptr_;

#ifndef APIENTRY
  #if defined(_WIN32)
    #define APIENTRY __stdcall
  #else
    #define APIENTRY
  #endif
#endif

#define GLFUNCS \
    X(GLuint, CreateShader, (GLenum t), (t)) \
    X(void,   ShaderSource, (GLuint s, GLsizei n, const GLchar_ * const *p, const GLint *l), (s,n,p,l)) \
    X(void,   CompileShader,(GLuint s), (s)) \
    X(void,   GetShaderiv,  (GLuint s, GLenum p, GLint *v), (s,p,v)) \
    X(void,   GetShaderInfoLog,(GLuint s, GLsizei b, GLsizei *l, GLchar_ *i), (s,b,l,i)) \
    X(void,   DeleteShader, (GLuint s), (s)) \
    X(GLuint, CreateProgram,(void), ()) \
    X(void,   AttachShader, (GLuint p, GLuint s), (p,s)) \
    X(void,   LinkProgram,  (GLuint p), (p)) \
    X(void,   GetProgramiv, (GLuint p, GLenum n, GLint *v), (p,n,v)) \
    X(void,   GetProgramInfoLog,(GLuint p, GLsizei b, GLsizei *l, GLchar_ *i), (p,b,l,i)) \
    X(void,   UseProgram,   (GLuint p), (p)) \
    X(GLint,  GetUniformLocation,(GLuint p, const GLchar_ *n), (p,n)) \
    X(void,   UniformMatrix4fv,(GLint l, GLsizei c, GLboolean t, const GLfloat *v), (l,c,t,v)) \
    X(void,   Uniform1f,    (GLint l, GLfloat v), (l,v)) \
    X(void,   GenVertexArrays,(GLsizei n, GLuint *a), (n,a)) \
    X(void,   BindVertexArray,(GLuint a), (a)) \
    X(void,   GenBuffers,   (GLsizei n, GLuint *b), (n,b)) \
    X(void,   BindBuffer,   (GLenum t, GLuint b), (t,b)) \
    X(void,   BufferData,   (GLenum t, GLsizeiptr_ s, const void *d, GLenum u), (t,s,d,u)) \
    X(void,   BufferSubData,(GLenum t, GLintptr_ o, GLsizeiptr_ s, const void *d), (t,o,s,d)) \
    X(void,   VertexAttribPointer,(GLuint i, GLint s, GLenum t, GLboolean n, GLsizei st, const void *p), (i,s,t,n,st,p)) \
    X(void,   EnableVertexAttribArray,(GLuint i), (i))

#define X(ret, name, params, args) typedef ret (APIENTRY *PFN_gl_##name) params; static PFN_gl_##name gl_##name;
GLFUNCS
#undef X

static void viz_load_gl_funcs(void) {
#define X(ret, name, params, args) gl_##name = (PFN_gl_##name)SDL_GL_GetProcAddress("gl" #name);
    GLFUNCS
#undef X
}

/* Redirect the modern names we use below to the loaded function
   pointers, so the code reads identically on both platforms. */
#define glCreateShader          gl_CreateShader
#define glShaderSource          gl_ShaderSource
#define glCompileShader         gl_CompileShader
#define glGetShaderiv           gl_GetShaderiv
#define glGetShaderInfoLog      gl_GetShaderInfoLog
#define glDeleteShader          gl_DeleteShader
#define glCreateProgram         gl_CreateProgram
#define glAttachShader          gl_AttachShader
#define glLinkProgram           gl_LinkProgram
#define glGetProgramiv          gl_GetProgramiv
#define glGetProgramInfoLog     gl_GetProgramInfoLog
#define glUseProgram            gl_UseProgram
#define glGetUniformLocation    gl_GetUniformLocation
#define glUniformMatrix4fv      gl_UniformMatrix4fv
#define glUniform1f             gl_Uniform1f
#define glGenVertexArrays       gl_GenVertexArrays
#define glBindVertexArray       gl_BindVertexArray
#define glGenBuffers            gl_GenBuffers
#define glBindBuffer            gl_BindBuffer
#define glBufferData            gl_BufferData
#define glBufferSubData         gl_BufferSubData
#define glVertexAttribPointer   gl_VertexAttribPointer
#define glEnableVertexAttribArray gl_EnableVertexAttribArray

#ifndef GL_ARRAY_BUFFER
  #define GL_ARRAY_BUFFER       0x8892
#endif
#ifndef GL_DYNAMIC_DRAW
  #define GL_DYNAMIC_DRAW       0x88E8
#endif
#ifndef GL_VERTEX_SHADER
  #define GL_VERTEX_SHADER      0x8B31
  #define GL_FRAGMENT_SHADER    0x8B30
#endif
#ifndef GL_COMPILE_STATUS
  #define GL_COMPILE_STATUS     0x8B81
  #define GL_LINK_STATUS        0x8B82
  #define GL_INFO_LOG_LENGTH    0x8B84
#endif
#ifndef GL_PROGRAM_POINT_SIZE
  #define GL_PROGRAM_POINT_SIZE 0x8642
#endif
#ifndef GL_MULTISAMPLE
  #define GL_MULTISAMPLE        0x809D
#endif

#else  /* __EMSCRIPTEN__ */
static void viz_load_gl_funcs(void) {}
#endif

/* =================================================================
 *  Matrix stack (replaces fixed-function glMatrixMode/glLoadIdentity
 *  /glPushMatrix/glPopMatrix/glMultMatrixf/glOrtho/gluPerspective/
 *  gluLookAt).  Column-major, like GL.
 * ================================================================= */

typedef float Mat4[16];

#define MAT_STACK_DEPTH 32
static Mat4 g_mv_stack[MAT_STACK_DEPTH];
static Mat4 g_pj_stack[MAT_STACK_DEPTH];
static int  g_mv_top = 0;
static int  g_pj_top = 0;
static int  g_mat_mode = GL_MODELVIEW;
static int  g_viewport[4] = { 0, 0, 1200, 820 };

static void mat_identity(Mat4 m) {
    memset(m, 0, sizeof(Mat4));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}
static void mat_copy(Mat4 dst, const Mat4 src) { memcpy(dst, src, sizeof(Mat4)); }
static void mat_mul(Mat4 out, const Mat4 a, const Mat4 b) {
    Mat4 r;
    for (int c = 0; c < 4; c++)
        for (int rw = 0; rw < 4; rw++)
            r[rw + c*4] =
                a[rw     ] * b[c*4    ] +
                a[rw +  4] * b[c*4 + 1] +
                a[rw +  8] * b[c*4 + 2] +
                a[rw + 12] * b[c*4 + 3];
    memcpy(out, r, sizeof(Mat4));
}

static float *mat_cur(void) {
    return (g_mat_mode == GL_PROJECTION) ? g_pj_stack[g_pj_top] : g_mv_stack[g_mv_top];
}

static void viz_matrix_mode(GLenum m)  { g_mat_mode = m; }
static void viz_load_identity(void)    { mat_identity(mat_cur()); }
static void viz_push_matrix(void) {
    if (g_mat_mode == GL_PROJECTION) {
        if (g_pj_top < MAT_STACK_DEPTH - 1) {
            mat_copy(g_pj_stack[g_pj_top + 1], g_pj_stack[g_pj_top]); g_pj_top++;
        }
    } else {
        if (g_mv_top < MAT_STACK_DEPTH - 1) {
            mat_copy(g_mv_stack[g_mv_top + 1], g_mv_stack[g_mv_top]); g_mv_top++;
        }
    }
}
static void viz_pop_matrix(void) {
    if (g_mat_mode == GL_PROJECTION) { if (g_pj_top > 0) g_pj_top--; }
    else                             { if (g_mv_top > 0) g_mv_top--; }
}
static void viz_mult_matrixf(const float *m) {
    Mat4 r, c;  mat_copy(c, mat_cur());  mat_mul(r, c, m);  mat_copy(mat_cur(), r);
}
static void viz_ortho(double l, double r, double b, double t, double zn, double zf) {
    Mat4 m = {0};
    m[0]  = (float)( 2.0 / (r - l));
    m[5]  = (float)( 2.0 / (t - b));
    m[10] = (float)(-2.0 / (zf - zn));
    m[12] = (float)(-(r + l) / (r - l));
    m[13] = (float)(-(t + b) / (t - b));
    m[14] = (float)(-(zf + zn) / (zf - zn));
    m[15] = 1.0f;
    viz_mult_matrixf(m);
}
static void viz_perspective(double fovy_deg, double aspect, double zn, double zf) {
    double f = 1.0 / tan(fovy_deg * 0.5 * M_PI / 180.0);
    Mat4 m = {0};
    m[0]  = (float)(f / aspect);
    m[5]  = (float) f;
    m[10] = (float)((zf + zn) / (zn - zf));
    m[11] = -1.0f;
    m[14] = (float)((2.0 * zf * zn) / (zn - zf));
    viz_mult_matrixf(m);
}
static void viz_lookat(double ex, double ey, double ez,
                       double cx, double cy, double cz,
                       double ux, double uy, double uz) {
    double fx = cx-ex, fy = cy-ey, fz = cz-ez;
    double fl = sqrt(fx*fx + fy*fy + fz*fz);  fx/=fl; fy/=fl; fz/=fl;
    double sx = fy*uz - fz*uy, sy = fz*ux - fx*uz, sz = fx*uy - fy*ux;
    double sl = sqrt(sx*sx + sy*sy + sz*sz);  sx/=sl; sy/=sl; sz/=sl;
    double upx = sy*fz - sz*fy, upy = sz*fx - sx*fz, upz = sx*fy - sy*fx;
    Mat4 m = {
        (float)sx, (float)upx, (float)-fx, 0,
        (float)sy, (float)upy, (float)-fy, 0,
        (float)sz, (float)upz, (float)-fz, 0,
        0,         0,          0,          1
    };
    viz_mult_matrixf(m);
    Mat4 t = { 1,0,0,0,  0,1,0,0,  0,0,1,0,  (float)-ex, (float)-ey, (float)-ez, 1 };
    viz_mult_matrixf(t);
}

static void viz_get_doublev(GLenum e, double *out) {
    const float *src = NULL;
    if      (e == GL_MODELVIEW_MATRIX)  src = g_mv_stack[g_mv_top];
    else if (e == GL_PROJECTION_MATRIX) src = g_pj_stack[g_pj_top];
    else return;
    for (int i = 0; i < 16; i++) out[i] = (double)src[i];
}

/* Project / unproject using the tracked matrices + g_viewport. */
static void mat_mul_v4(const Mat4 m, const double v[4], double r[4]) {
    for (int i = 0; i < 4; i++)
        r[i] = (double)m[i]*v[0] + (double)m[i+4]*v[1] + (double)m[i+8]*v[2] + (double)m[i+12]*v[3];
}
static int  mat_mul_dd(const double a[16], const double b[16], double r[16]) {
    double o[16];
    for (int c = 0; c < 4; c++)
        for (int rw = 0; rw < 4; rw++)
            o[rw + c*4] = a[rw]*b[c*4] + a[rw+4]*b[c*4+1] + a[rw+8]*b[c*4+2] + a[rw+12]*b[c*4+3];
    memcpy(r, o, sizeof o);
    return 1;
}
static int mat_invert_d(const double m[16], double inv[16]) {
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
    det = 1.0 / det;
    for (int i = 0; i < 16; i++) inv[i] *= det;
    return 1;
}

static int viz_project(double ox, double oy, double oz,
                       const double mv[16], const double pj[16], const int vp[4],
                       double *wx, double *wy, double *wz) {
    Mat4 mvf, pjf;
    for (int i = 0; i < 16; i++) { mvf[i] = (float)mv[i]; pjf[i] = (float)pj[i]; }
    double v[4] = {ox, oy, oz, 1.0}, t[4], c[4];
    mat_mul_v4(mvf, v, t);
    mat_mul_v4(pjf, t, c);
    if (c[3] == 0) return 0;
    c[0]/=c[3]; c[1]/=c[3]; c[2]/=c[3];
    *wx = vp[0] + (c[0]*0.5 + 0.5) * vp[2];
    *wy = vp[1] + (c[1]*0.5 + 0.5) * vp[3];
    *wz = c[2]*0.5 + 0.5;
    return 1;
}

static int viz_unproject(double wx, double wy, double wz,
                         const double mv[16], const double pj[16], const int vp[4],
                         double *ox, double *oy, double *oz) {
    double mvp[16], inv[16];
    mat_mul_dd(pj, mv, mvp);
    if (!mat_invert_d(mvp, inv)) return 0;
    double v[4] = {
        (wx - vp[0]) / vp[2] * 2.0 - 1.0,
        (wy - vp[1]) / vp[3] * 2.0 - 1.0,
        wz * 2.0 - 1.0,
        1.0
    };
    double r[4];
    {   double tmp[4];
        for (int i = 0; i < 4; i++)
            tmp[i] = inv[i]*v[0] + inv[i+4]*v[1] + inv[i+8]*v[2] + inv[i+12]*v[3];
        memcpy(r, tmp, sizeof r);
    }
    if (r[3] == 0) return 0;
    *ox = r[0]/r[3]; *oy = r[1]/r[3]; *oz = r[2]/r[3];
    return 1;
}

/* =================================================================
 *  Immediate-mode batcher.  Accumulates vertices between
 *  viz_begin() / viz_end() then draws with a single shader.
 * ================================================================= */

typedef struct { float x, y, z, r, g, b, a; } ImmVert;

#define IMM_MAX_VERTS 262144
static ImmVert g_verts[IMM_MAX_VERTS];
static int     g_nvert = 0;
static GLenum  g_imm_mode = 0;
static float   g_cur_r = 1, g_cur_g = 1, g_cur_b = 1, g_cur_a = 1;
static float   g_line_width = 1.0f;
static float   g_point_size = 1.0f;

static GLuint  g_prog = 0;
static GLint   g_uni_mvp = -1, g_uni_pointsize = -1;
static GLuint  g_vao = 0, g_vbo = 0;

static GLuint viz_compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof log, &n, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
    }
    return s;
}

static void viz_init_gl(void) {
    static const char *vert_src =
#ifdef __EMSCRIPTEN__
        "#version 300 es\n"
        "precision highp float;\n"
#else
        "#version 330 core\n"
#endif
        "layout(location=0) in vec3 a_pos;\n"
        "layout(location=1) in vec4 a_col;\n"
        "uniform mat4 u_mvp;\n"
        "uniform float u_pointsize;\n"
        "out vec4 v_col;\n"
        "void main() {\n"
        "  gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
        "  gl_PointSize = u_pointsize;\n"
        "  v_col = a_col;\n"
        "}\n";
    static const char *frag_src =
#ifdef __EMSCRIPTEN__
        "#version 300 es\n"
        "precision highp float;\n"
#else
        "#version 330 core\n"
#endif
        "in vec4 v_col;\n"
        "out vec4 o_col;\n"
        "void main() { o_col = v_col; }\n";

    GLuint vs = viz_compile(GL_VERTEX_SHADER, vert_src);
    GLuint fs = viz_compile(GL_FRAGMENT_SHADER, frag_src);
    g_prog = glCreateProgram();
    glAttachShader(g_prog, vs);
    glAttachShader(g_prog, fs);
    glLinkProgram(g_prog);
    GLint ok = 0; glGetProgramiv(g_prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetProgramInfoLog(g_prog, sizeof log, &n, log);
        fprintf(stderr, "program link error:\n%s\n", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    g_uni_mvp       = glGetUniformLocation(g_prog, "u_mvp");
    g_uni_pointsize = glGetUniformLocation(g_prog, "u_pointsize");

    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof g_verts, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ImmVert), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(ImmVert), (void*)(3 * sizeof(float)));

#ifndef __EMSCRIPTEN__
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);
#endif

    mat_identity(g_mv_stack[0]);
    mat_identity(g_pj_stack[0]);
}

static void viz_flush(GLenum mode, int nv) {
    if (nv == 0 || !g_prog) return;
    Mat4 mvp;  mat_mul(mvp, g_pj_stack[g_pj_top], g_mv_stack[g_mv_top]);
    glUseProgram(g_prog);
    glUniformMatrix4fv(g_uni_mvp, 1, GL_FALSE, mvp);
    glUniform1f(g_uni_pointsize, g_point_size);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(nv * sizeof(ImmVert)), g_verts);
    glLineWidth(g_line_width);
    glDrawArrays(mode, 0, nv);
}

static void viz_begin(GLenum mode) { g_imm_mode = mode; g_nvert = 0; }
static void viz_color3f(float r, float g, float b) { g_cur_r=r; g_cur_g=g; g_cur_b=b; g_cur_a=1; }
static void viz_color4f(float r, float g, float b, float a) { g_cur_r=r; g_cur_g=g; g_cur_b=b; g_cur_a=a; }
static void viz_vertex3f(float x, float y, float z) {
    if (g_nvert >= IMM_MAX_VERTS) return;
    ImmVert *v = &g_verts[g_nvert++];
    v->x = x; v->y = y; v->z = z;
    v->r = g_cur_r; v->g = g_cur_g; v->b = g_cur_b; v->a = g_cur_a;
}
static void viz_vertex2f(float x, float y) { viz_vertex3f(x, y, 0.0f); }

static void viz_end(void) {
    int n = g_nvert;
    GLenum mode = g_imm_mode;
    if (mode == GL_QUADS) {
        int q = n / 4;
        if (q == 0) return;
        /* expand 0,1,2,3 -> 0,1,2, 0,2,3 */
        static ImmVert tmp[IMM_MAX_VERTS];
        int out = 0;
        for (int i = 0; i < q; i++) {
            int b = i * 4;
            tmp[out++] = g_verts[b+0];
            tmp[out++] = g_verts[b+1];
            tmp[out++] = g_verts[b+2];
            tmp[out++] = g_verts[b+0];
            tmp[out++] = g_verts[b+2];
            tmp[out++] = g_verts[b+3];
        }
        memcpy(g_verts, tmp, out * sizeof(ImmVert));
        viz_flush(GL_TRIANGLES, out);
    } else if (mode == GL_QUAD_STRIP) {
        viz_flush(GL_TRIANGLE_STRIP, n);
    } else if (mode == GL_POLYGON) {
        viz_flush(GL_TRIANGLE_FAN, n);
    } else {
        viz_flush(mode, n);
    }
    g_nvert = 0;
}

static void viz_line_width(float w) { g_line_width = w > 0 ? w : 1.0f; }
static void viz_point_size(float s) { g_point_size = s > 0 ? s : 1.0f; }

/* enable / disable with a filter for the legacy fixed-function caps. */
static void viz_enable(GLenum cap) {
    switch (cap) {
    case GL_LIGHTING: case GL_LIGHT0:
    case GL_LINE_SMOOTH: case GL_POINT_SMOOTH: case GL_LINE_STIPPLE:
        return;
    default: glEnable(cap);
    }
}
static void viz_disable(GLenum cap) {
    switch (cap) {
    case GL_LIGHTING: case GL_LIGHT0:
    case GL_LINE_SMOOTH: case GL_POINT_SMOOTH: case GL_LINE_STIPPLE:
        return;
    default: glDisable(cap);
    }
}

/* Viewport wrapper tracks the current viewport so gluProject/UnProject
   and 2-D text rendering can read it without calling back into GL. */
static void viz_viewport(int x, int y, int w, int h) {
    g_viewport[0] = x; g_viewport[1] = y; g_viewport[2] = w; g_viewport[3] = h;
    glViewport(x, y, w, h);
}

/* =================================================================
 *  gluSphere replacement (triangle strips).  Old code called this
 *  through the GLU quadric API; we keep the signatures via macros.
 * ================================================================= */
typedef int GLUquadric;
#ifndef GLU_FILL
  #define GLU_FILL    0
  #define GLU_SMOOTH  0
#endif
static GLUquadric *viz_new_quadric(void) { static int q; return &q; }
static void viz_sphere(GLUquadric *q, double r, int slices, int stacks) {
    (void)q;
    for (int i = 0; i < stacks; i++) {
        double p0 = M_PI * ((double)i       / stacks - 0.5);
        double p1 = M_PI * ((double)(i + 1) / stacks - 0.5);
        double cp0 = cos(p0), sp0 = sin(p0);
        double cp1 = cos(p1), sp1 = sin(p1);
        viz_begin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= slices; j++) {
            double th = 2.0 * M_PI * j / slices;
            double ct = cos(th), st = sin(th);
            viz_vertex3f((float)(r*cp1*ct), (float)(r*sp1), (float)(r*cp1*st));
            viz_vertex3f((float)(r*cp0*ct), (float)(r*sp0), (float)(r*cp0*st));
        }
        viz_end();
    }
}

/* =================================================================
 *  Text rendering via stb_easy_font.  Converts stb's quads into
 *  triangles fed through viz_* (so text obeys the current MVP).
 * ================================================================= */

static void viz_text_2d_px(int x_from_left, int y_from_top, const char *s) {
    static char buf[99999];
    int nq = stb_easy_font_print((float)x_from_left, (float)y_from_top,
                                 (char*)s, NULL, buf, (int)sizeof buf);
    /* stb emits quads: 4 verts each, 16 bytes per vert
       (float x,y,z then uchar[4] color).  y grows downwards. */
    const char *p = buf;
    viz_begin(GL_QUADS);
    for (int i = 0; i < nq; i++) {
        for (int v = 0; v < 4; v++) {
            float qx, qy, qz;
            memcpy(&qx, p + 0, 4);
            memcpy(&qy, p + 4, 4);
            memcpy(&qz, p + 8, 4);
            p += 16;
            viz_vertex3f(qx, (float)win_h - qy, qz);  /* flip y */
        }
    }
    viz_end();
}

static void gl_text_2d(int x, int y_from_top, const char *s) {
    viz_text_2d_px(x, y_from_top, s);
}

static void gl_text_3d_xyz(float wx, float wy, float wz, const char *s);
/* Forward -- defined below once Vec3 is available. */

/* =================================================================
 *  Macro redirection -- keep the app code identical to the old
 *  fixed-function GL by routing legacy calls to viz_* wrappers.
 * ================================================================= */

#define glBegin(m)              viz_begin(m)
#define glEnd()                 viz_end()
#define glVertex3f(x,y,z)       viz_vertex3f((float)(x),(float)(y),(float)(z))
#define glVertex3d(x,y,z)       viz_vertex3f((float)(x),(float)(y),(float)(z))
#define glVertex2f(x,y)         viz_vertex2f((float)(x),(float)(y))
#define glVertex2i(x,y)         viz_vertex2f((float)(x),(float)(y))
#define glColor3f(r,g,b)        viz_color3f((float)(r),(float)(g),(float)(b))
#define glColor4f(r,g,b,a)      viz_color4f((float)(r),(float)(g),(float)(b),(float)(a))
#define glNormal3f(x,y,z)       ((void)0)
#define glNormal3d(x,y,z)       ((void)0)
#define glMatrixMode(m)         viz_matrix_mode(m)
#define glLoadIdentity()        viz_load_identity()
#define glPushMatrix()          viz_push_matrix()
#define glPopMatrix()           viz_pop_matrix()
#define glMultMatrixf(m)        viz_mult_matrixf(m)
#define glOrtho(l,r,b,t,zn,zf)  viz_ortho((double)(l),(double)(r),(double)(b),(double)(t),(double)(zn),(double)(zf))
#define glEnable(c)             viz_enable(c)
#define glDisable(c)            viz_disable(c)
#define glLightfv(a,b,c)        ((void)0)
#define glMaterialfv(a,b,c)     ((void)0)
#define glMaterialf(a,b,c)      ((void)0)
#define glHint(a,b)             ((void)0)
#define glGetDoublev(e,p)       viz_get_doublev((e),(p))
#define glViewport(x,y,w,h)     viz_viewport((x),(y),(w),(h))
#define glLineWidth(w)          viz_line_width((float)(w))
#define glPointSize(s)          viz_point_size((float)(s))

/* Viewport read through glGetIntegerv.  Only used for GL_VIEWPORT
   in this codebase -- forward to tracked state. */
static void viz_get_viewport_ints(int *p) {
    p[0] = g_viewport[0]; p[1] = g_viewport[1];
    p[2] = g_viewport[2]; p[3] = g_viewport[3];
}
#define glGetIntegerv(e,p)      do { if ((e) == 0x0BA2 /*GL_VIEWPORT*/) viz_get_viewport_ints((p)); } while (0)

#define gluPerspective(f,a,n,x) viz_perspective((double)(f),(double)(a),(double)(n),(double)(x))
#define gluLookAt(ex,ey,ez,cx,cy,cz,ux,uy,uz) viz_lookat((double)(ex),(double)(ey),(double)(ez),(double)(cx),(double)(cy),(double)(cz),(double)(ux),(double)(uy),(double)(uz))
#define gluProject(a,b,c,mv,pj,vp,x,y,z)   viz_project((double)(a),(double)(b),(double)(c),(mv),(pj),(vp),(x),(y),(z))
#define gluUnProject(a,b,c,mv,pj,vp,x,y,z) viz_unproject((double)(a),(double)(b),(double)(c),(mv),(pj),(vp),(x),(y),(z))
#define gluSphere(q,r,sl,st)    viz_sphere((q),(double)(r),(sl),(st))
#define gluNewQuadric()         viz_new_quadric()
#define gluQuadricDrawStyle(q,s) ((void)0)
#define gluQuadricNormals(q,m)  ((void)0)

/* PLAT_INVALIDATE is a no-op now (we render every frame). */
#define PLAT_INVALIDATE()       ((void)0)

enum { KEY_ESCAPE = 1, KEY_R = 2, KEY_LEFT = 3, KEY_RIGHT = 4 };

static int g_should_quit = 0;

/* Forward decls -- real definitions at the bottom of the file. */
static void plat_swap(void);

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

static void init_font(void) {}  /* stb_easy_font needs no init */

/* 3-D text: project to screen, draw as 2-D quads in ortho. */
static void gl_text_3d(Vec3 p, const char *s) {
    double mv[16], pj[16];
    viz_get_doublev(GL_MODELVIEW_MATRIX,  mv);
    viz_get_doublev(GL_PROJECTION_MATRIX, pj);
    int vp[4]; for (int i = 0; i < 4; i++) vp[i] = g_viewport[i];
    double sx, sy, sz;
    if (!viz_project(p.x, p.y, p.z, mv, pj, vp, &sx, &sy, &sz)) return;
    if (sz < 0.0 || sz > 1.0) return;

    /* Bracket: switch to full-window ortho, draw, restore. */
    viz_matrix_mode(GL_PROJECTION);
    viz_push_matrix();
    viz_load_identity();
    viz_ortho(0, win_w, 0, win_h, -1, 1);
    viz_matrix_mode(GL_MODELVIEW);
    viz_push_matrix();
    viz_load_identity();
    int vp_save[4] = { g_viewport[0], g_viewport[1], g_viewport[2], g_viewport[3] };
    viz_viewport(0, 0, win_w, win_h);

    int y_from_top = win_h - (int)sy;
    gl_text_2d((int)sx, y_from_top, s);

    viz_viewport(vp_save[0], vp_save[1], vp_save[2], vp_save[3]);
    viz_matrix_mode(GL_PROJECTION);
    viz_pop_matrix();
    viz_matrix_mode(GL_MODELVIEW);
    viz_pop_matrix();
}

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
static int slider_x0(void) { return 100; }
static int slider_x1(void) { return win_w - 20; }
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
}

/* LHS sidebar with two vertical rotation-amount sliders. */
static void draw_sidebar(void) {
    int side_y0 = SLIDER_H;
    int side_h  = win_h - SLIDER_H - PLOT_H;

    /* Background. */
    fill_rect_2d(0, side_y0, SIDEBAR_W, side_h, 0.12f, 0.12f, 0.16f, 1.0f);

    /* Right-edge separator. */
    fill_rect_2d(SIDEBAR_W - 1, side_y0, 1, side_h, 0.28f, 0.28f, 0.34f, 1.0f);

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
    /* Translucent dark fill so the wire + arcs on the far side show
       through.  Lighting is gone in the modern-GL port; a flat low-
       alpha colour reads as a see-through sphere just fine.          */
    static GLUquadric *q = NULL;
    if (!q) q = gluNewQuadric();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.10f, 0.10f, 0.13f, 0.35f);
    gluSphere(q, 0.992, 40, 20);
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
    {
        const char *hud = "LMB orbit   RMB trackball-rotate box   Wheel zoom   R reset   Esc quit";
        int hud_w = stb_easy_font_width((char*)hud);
        int view_left  = SIDEBAR_W;
        int view_right = win_w - INSET_W - 2 * INSET_PAD;
        int cx = view_left + (view_right - view_left) / 2 - hud_w / 2;
        if (cx < view_left + 6) cx = view_left + 6;
        gl_text_2d(cx, SLIDER_H + 18, hud);
    }

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
        g_should_quit = 1;
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
 *  Platform layer: SDL2 windowing + main loop (desktop and web).
 * ================================================================= */

static SDL_Window    *g_sdl_win = NULL;
static SDL_GLContext  g_sdl_ctx = NULL;

static int sdl_button_to_my(int b) {
    if (b == SDL_BUTTON_LEFT)  return 0;
    if (b == SDL_BUTTON_RIGHT) return 1;
    return -1;
}

static int sdl_key_to_my(SDL_Keycode k) {
    switch (k) {
    case SDLK_ESCAPE: return KEY_ESCAPE;
    case SDLK_r:      return KEY_R;
    case SDLK_LEFT:   return KEY_LEFT;
    case SDLK_RIGHT:  return KEY_RIGHT;
    }
    return 0;
}

static void plat_poll_events(void) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_QUIT:
            g_should_quit = 1;
            break;
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                e.window.event == SDL_WINDOWEVENT_RESIZED) {
                on_resize(e.window.data1, e.window.data2);
            }
            break;
        case SDL_MOUSEBUTTONDOWN: {
            int b = sdl_button_to_my(e.button.button);
            if (b >= 0) on_mouse_down(b, e.button.x, e.button.y);
            break;
        }
        case SDL_MOUSEBUTTONUP: {
            int b = sdl_button_to_my(e.button.button);
            if (b >= 0) on_mouse_up(b);
            break;
        }
        case SDL_MOUSEMOTION:
            on_mouse_move(e.motion.x, e.motion.y);
            break;
        case SDL_MOUSEWHEEL: {
            int mx = 0, my = 0;
            SDL_GetMouseState(&mx, &my);
            on_mouse_wheel((float)e.wheel.y, mx, my);
            break;
        }
        case SDL_KEYDOWN: {
            int k = sdl_key_to_my(e.key.keysym.sym);
            if (k) on_key_down(k);
            break;
        }
        }
    }
}

static void plat_frame(void) {
    plat_poll_events();
    render();
    SDL_GL_SwapWindow(g_sdl_win);
}

static void plat_swap(void) { /* SDL_GL_SwapWindow is called in plat_frame */ }

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

#ifdef __EMSCRIPTEN__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#endif
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    g_sdl_win = SDL_CreateWindow(
        "Edge-Edge Motion  |  Signed Separation",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        win_w, win_h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!g_sdl_win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }
    g_sdl_ctx = SDL_GL_CreateContext(g_sdl_win);
    if (!g_sdl_ctx) {
        fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_MakeCurrent(g_sdl_win, g_sdl_ctx);
    SDL_GL_SetSwapInterval(1);

    viz_load_gl_funcs();
    viz_init_gl();
    init_font();
    reset_scene();
    glViewport(0, 0, win_w, win_h);

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(plat_frame, 0, 1);
#else
    while (!g_should_quit) plat_frame();
    SDL_GL_DeleteContext(g_sdl_ctx);
    SDL_DestroyWindow(g_sdl_win);
    SDL_Quit();
#endif
    return 0;
}
