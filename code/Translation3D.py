import ctypes
import sys
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glm as glm

# Define Shaders
vertexShader = """
attribute vec4 color;
attribute vec3 position;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 transform;
varying vec4 v_color;
void main()
{
  gl_Position = transform * vec4(position, 1.0);
  v_color= color;
}
"""

fragmentShader = """
varying vec4 v_color;
void main()
{
  gl_FragColor = v_color;
}
"""


# Build data

tempData = (
    (0.5, 0.5, -0.5),
    (0.5, 0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, -0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, 0.5, 0.5),
    (0.5, -0.5, -0.5),
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, -0.5, -0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (0.5, -0.5, -0.5),
)

tempColor = (
    (0.714, 0.505, 0.345, 1.0),  # t
    (0.714, 0.505, 0.345, 1.0),  # t
    (0.783, 0.290, 0.734, 1.0),  # t
    (0.714, 0.505, 0.345, 1.0),  # t
    # (0.714, 0.505, 0.345, 1.0),  # t
    # (0.783, 0.290, 0.734, 1.0),  # t
    (0.997, 0.513, 0.064, 1.0),  # d
    (0.945, 0.719, 0.592, 1.0),  # d
    (0.543, 0.021, 0.978, 1.0),  # d
    (0.673, 0.211, 0.457, 1.0),  # d
    # (0.820, 0.883, 0.371, 1.0),  # d
    # (0.982, 0.099, 0.879, 1.0),  # d
    (0.722, 0.645, 0.174, 1.0),  # s
    (0.302, 0.455, 0.848, 1.0),  # s
    (0.225, 0.587, 0.040, 1.0),  # s
    (0.517, 0.713, 0.338, 1.0),  # s
    # (0.053, 0.959, 0.120, 1.0),  # s
    # (0.393, 0.621, 0.362, 1.0),  # s
    (0.997, 0.513, 0.064, 1.0),  # f
    (0.945, 0.719, 0.592, 1.0),  # f
    (0.543, 0.021, 0.978, 1.0),  # f
    (0.997, 0.513, 0.064, 1.0),  # f
    # (0.945, 0.719, 0.592, 1.0),  # f
    # (0.945, 0.719, 0.592, 1.0),  # f
    (0.583, 0.771, 0.014, 1.0),  # b
    (0.609, 0.115, 0.436, 1.0),  # b
    (0.327, 0.483, 0.844, 1.0),  # b
    (0.014, 0.184, 0.576, 1.0),  # b
    # (0.771, 0.328, 0.970, 1.0),  # b
    # (0.406, 0.615, 0.116, 1.0),  # b
    (0.559, 0.861, 0.639, 1.0),  # h
    (0.559, 0.861, 0.639, 1.0),  # h
    (0.195, 0.548, 0.859, 1.0),  # h
    (0.559, 0.861, 0.639, 1.0),  # h
    # (0.195, 0.548, 0.859, 1.0),  # h
    # (0.559, 0.861, 0.639, 1.0),  # h
)

vertex_data = np.zeros(
    int(len(tempData)), [("position", np.float32, 3), ("colors", np.float32, 4)]
)
vertex_data["position"] = tempData
vertex_data["colors"] = tempColor

indicesData = np.array(
    [
        0,
        1,
        2,
        1,
        2,
        3,
        4,
        5,
        6,
        5,
        6,
        7,
        8,
        9,
        10,
        9,
        10,
        11,
        12,
        13,
        14,
        13,
        14,
        15,
        16,
        17,
        18,
        17,
        18,
        19,
        20,
        21,
        22,
        21,
        22,
        23,
    ],
    dtype=np.int32,
)


def compileShader(source, type):
    shader = gl.glCreateShader(type)
    gl.glShaderSource(shader, source)

    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        print(error)
        raise RuntimeError("{source} shader compilation error")
    return shader


def createProgram(vertex, fragment):
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)

    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        print(gl.glGetProgramInfoLog(program))
        raise RuntimeError("Error Linking program")

    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    return program


def init():
    global program
    global data

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glLoadIdentity()

    program = createProgram(
        compileShader(vertexShader, gl.GL_VERTEX_SHADER),
        compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER),
    )

    # Use Program
    gl.glUseProgram(program)

    vertexBuffer = gl.glGenBuffers(1)
    indicesBuffer = gl.glGenBuffers(1)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indicesBuffer)

    stride = vertex_data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    offset = ctypes.c_void_p(vertex_data.dtype["position"].itemsize)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

    # view = glm.lookAt(
    #     glm.vec3(1.0, 1.0, 1.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0)
    # )
    # loc = gl.glGetUniformLocation(program, "view")
    # gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(view))

    # projection = glm.perspective(glm.radians(45.0), 1.0, 0.2, 0.8)
    # loc = gl.glGetUniformLocation(program, "transform")
    # gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(projection))

    transform = glm.mat4(1)
    transform = glm.rotate(transform, glm.radians(90), glm.vec3(1.0, 1.0, 1.0))
    loc = gl.glGetUniformLocation(program, "transform")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, glm.value_ptr(transform))

    # print(projection)
    # print(" ")
    # print(view)
    # print(" ")
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW
    )

    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, indicesData.nbytes, indicesData, gl.GL_STATIC_DRAW
    )


def render():

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glDrawElements(gl.GL_TRIANGLES, len(indicesData), gl.GL_UNSIGNED_INT, None)
    gl.glFlush()
    glut.glutSwapBuffers()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b"\x1b":
        sys.exit()


glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)  # type: ignore #ignore
glut.glutCreateWindow("Translation in 3D")
glut.glutReshapeWindow(512, 512)
glut.glutReshapeFunc(reshape)
init()
glut.glutDisplayFunc(render)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)
glut.glutMainLoop()
