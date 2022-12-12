import os
import sys
import ctypes
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut

vertexShaderCode = """
    attribute vec3 position;
    attribute vec4 color;
    varying vec4 vColor;

    void main(){
        gl_Position = vec4(position, 1.0);
        vColor = color;
    }
    """

fragmentShaderCode = """
    varying vec4 vColor;
    void main(){
        gl_FragColor = vColor;
    }
    """

# function to request and compiler shader slots from GPU
def createShader(source, type):
    # request shader
    shader = gl.glCreateShader(type)

    # set shader source using the code
    gl.glShaderSource(shader, source)

    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        print(error)
        raise RuntimeError(f"{source} shader compilation error")

    return shader

# func to build and activate program
def createProgram(vertex, fragment):
    program = gl.glCreateProgram()

    # attach shader objects to the program
    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)

    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        print(gl.glGetProgramInfoLog(program))
        raise RuntimeError('Linking error')

    # Get rid of shaders (no more needed)
    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    return program


# -- Building Data -- 
data = np.zeros(16, [("position", np.float32, 3),
                    ("color",    np.float32, 4)])
data['position'] = ((-0.5,0.5,0.5),
                    (0.5,0.5,0.5), 
                    (-0.5,-0.5,0.5), 
                    (0.5,-0.5,0.5), 
                    (-0.3,0.7,-0.5),
                    (0.7,0.7,-0.5), 
                    (-0.3,-0.3,-0.5), 
                    (0.7,-0.3,-0.5),
                    (-0.5,-0.5,0.5),
                    (-0.3,-0.3,-0.5),
                    (0.5,-0.5,0.5),
                    (0.7,-0.3,-0.5),
                    (-0.5,0.5,0.5),
                    (-0.3,0.7,-0.5),
                    (-0.5,-0.5,0.5),
                    (-0.3,-0.3,-0.5),
                    )
data['color'] = (
                (1.0,0.0,0.0, 1.0),
                (1.0,0.0,0.0, 1.0),
                (1.0,0.0,0.0, 1.0),
                (1.0,0.0,0.0, 1.0),
                (1.0,1.0,0.0, 1.0),
                (1.0,1.0,0.0, 1.0),
                (1.0,1.0,0.0, 1.0),
                (1.0,1.0,0.0, 1.0),
                (1.0,1.0,1.0, 1.0),
                (1.0,1.0,1.0, 1.0),
                (1.0,1.0,1.0, 1.0),
                (1.0,1.0,1.0, 1.0),
                (1.0,0.0,1.0, 1.0),
                (1.0,0.0,1.0, 1.0),
                (1.0,0.0,1.0, 1.0),
                (1.0,0.0,1.0, 1.0),
                )
indicesData = np.array([0,1,2,1,2,3,4,5,6,5,6,7,8,9,10,9,10,11, 12, 13,14,13, 14, 15], np.int32)

# initialization function
def initialize():
    global program
    global data

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glLoadIdentity()

    program = createProgram(
        createShader(vertexShaderCode, gl.GL_VERTEX_SHADER),
        createShader(fragmentShaderCode, gl.GL_FRAGMENT_SHADER),
    )

    # make program the default program
    gl.glUseProgram(program)

    buffer = gl.glGenBuffers(1)
    indicesBuffer = gl.glGenBuffers(1)

    # make these buffer the default one
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indicesBuffer)

    # bind the position attribute
    stride = data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    offset = ctypes.c_void_p(data.dtype["position"].itemsize)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)
    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indicesData.nbytes, indicesData, gl.GL_STATIC_DRAW)



def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawElements(gl.GL_TRIANGLES, len(indicesData),gl.GL_UNSIGNED_INT, None)
    gl.glFlush()
    glut.glutSwapBuffers()


def reshape(width,height):
    gl.glViewport(0, 0, width, height)


def keyboard( key, x, y):
    if key == b'\x1b':
        os._exit(1)

# GLUT init
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
glut.glutCreateWindow('Graphics Window')
glut.glutReshapeWindow(800,800)
glut.glutReshapeFunc(reshape)

initialize()

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()