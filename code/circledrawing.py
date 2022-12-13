import os
import sys
import ctypes
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut

vertexShaderCode = """
    attribute vec3 position;
    void main(){
        gl_Position = vec4(position, 1.0);
    }
    """

fragmentShaderCode = """
    uniform vec4 vColor;
    void main(){
        gl_FragColor = vColor;
    }
    """

# -- Building Data -- 
def circleDrawing():
    data = []

    if len(sys.argv) == 6:
        radius = int(sys.argv[1])
        center = [int(sys.argv[2]), int(sys.argv[3])]
        resolution = [int(sys.argv[4]), int(sys.argv[5])]

        if radius > 0:
            xValue = 0
            yValue = radius

            Pk = 1 - radius
            while (xValue <= yValue):

                data.append([xValue, yValue, 1.0])

                if (Pk < 0):
                    Pk = Pk + 2 * xValue + 3
                else:
                    Pk = Pk + 2 * (xValue - yValue) + 5
                    yValue = yValue - 1
                
                xValue = xValue + 1

            data = generateOtherPoints(data, center)
    else:
        raise Exception("Arguments do not match. Correctly Enter Parameters in format : [radius, center X, center Y, resoultion X and resoultion Y]")

    return data, resolution


def generateOtherPoints(data, center):
    circlePoints = []
    for point in data:
        circlePoints.append([point[0] + center[0], point[1] + center[1], point[2]])
        circlePoints.append([-point[0] + center[0], point[1] + center[1], point[2]])
        circlePoints.append([point[0] + center[0], -point[1] + center[1], point[2]])
        circlePoints.append([-point[0] + center[0], -point[1] + center[1], point[2]])
        circlePoints.append([point[1] + center[0], point[0] + center[1], point[2]])
        circlePoints.append([-point[1] + center[0], point[0] + center[1], point[2]])
        circlePoints.append([point[1] + center[0], -point[0] + center[1], point[2]])
        circlePoints.append([-point[1] + center[0], -point[0] + center[1], point[2]])


    return circlePoints

def tonormalized(coordinates, resolution):
    for coordinate in (coordinates):
        coordinate[0] = coordinate[0] * 2 / (resolution[0])
        coordinate[1] = coordinate[1] * 2 / (resolution[1])

    return np.array(coordinates, dtype = np.float32)
    
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

# initialization function
def initialize():
    global program
    global data

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glLoadIdentity()

    program = createProgram(
        createShader(vertexShaderCode, gl.GL_VERTEX_SHADER),
        createShader(fragmentShaderCode, gl.GL_FRAGMENT_SHADER),
    )

    # make program the default program
    gl.glUseProgram(program)

    buffer = gl.glGenBuffers(1)

    # make these buffer the default one
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

    # bind the position attribute
    stride = data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    loc = gl.glGetUniformLocation(program, "vColor")
    gl.glUniform4fv(loc, 1, [1.0,1.0,1.0,1.0])

    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_POINTS, 0, data.shape[0])
    glut.glutSwapBuffers()


def reshape(width,height):
    gl.glViewport(0, 0, width, height)


def keyboard( key, x, y):
    if key == b'\x1b':
        os._exit(1)

# GLUT init
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutCreateWindow('Graphics Window')
glut.glutReshapeWindow(800,800)
glut.glutReshapeFunc(reshape)

data, resolution = circleDrawing()
data = tonormalized(data, resolution)
initialize()

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()