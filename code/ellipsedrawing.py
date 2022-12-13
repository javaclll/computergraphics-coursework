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
def ellipseDrawing():
    data = []

    if len(sys.argv) >= 7:
        radius = [int(sys.argv[1]), int(sys.argv[2])]
        center = [int(sys.argv[3]), int(sys.argv[4])]
        resolution = [int(sys.argv[5]), int(sys.argv[6])]

        rX, rY = radius
        rXSquared = rX ** 2
        rYSquared = rY ** 2

        xValue = 0
        yValue = rY


        P1k = rYSquared + (1/4) * (rXSquared) - (rXSquared * rY)
        delX = 2 * rYSquared * xValue
        delY = 2 * rXSquared * yValue

        while (delX < delY):

            data.append([xValue, yValue, 1.0])
            
            xValue = xValue + 1

            if (P1k < 0):
                delX = delX + 2 * rYSquared
                P1k = P1k + delX +  rYSquared
            else:
                yValue = yValue - 1
                delX = delX + 2 * rYSquared
                delY = delY - 2 * rXSquared
                P1k = P1k + delX - delY + rYSquared
            
        P2k = rYSquared * (xValue + 1/2) * (xValue + 1/2) + rXSquared * (yValue - 1) * (yValue - 1) - rXSquared*rYSquared

        while (yValue >= 0):

            data.append([xValue, yValue, 1.0])
            
            yValue = yValue - 1

            if P2k < 0:
                xValue = xValue + 1
                delX = delX + (2 * rYSquared)
                delY = delY - (2 * rXSquared)
                P2k = P2k + delX - delY + rXSquared
            else:
                delY = delY - 2 * rXSquared
                P2k = P2k + rXSquared - delY

        data = generateOtherPoints(data, center)
    else:
        raise Exception("Arguments do not match. Correctly Enter Parameters in format : [ major radius X, minor radius Y, center X, center Y, resoultion X and resoultion Y]")

    return data, resolution


def generateOtherPoints(data, center):
    ellipsePoints = []
    for point in data:
        ellipsePoints.append([point[0] + center[0], point[1] + center[1], point[2]])
        ellipsePoints.append([-point[0] + center[0], point[1] + center[1], point[2]])
        ellipsePoints.append([point[0] + center[0], -point[1] + center[1], point[2]])
        ellipsePoints.append([-point[0] + center[0], -point[1] + center[1], point[2]])

    return ellipsePoints
    

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

data, resolution = ellipseDrawing()
data = tonormalized(data, resolution)
initialize()

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()