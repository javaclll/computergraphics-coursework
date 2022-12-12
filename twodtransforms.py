import os
import sys
import ctypes
import numpy as np
import math
import OpenGL.GL as gl
import OpenGL.GLUT as glut

vertexShaderCode = """
    attribute vec3 position;
    uniform mat4 transformMatrix;  
    uniform mat4 translate;

    mat4 retranslate;  

    void main(){

        retranslate = translate;
        retranslate[0][3] *= -1.0;
        retranslate[1][3] *= -1.0;

        gl_Position = retranslate * transformMatrix * translate * vec4(position, 1.0);
    }
    """

fragmentShaderCode = """
    uniform vec4 vColor;
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
def buildData():

    if len(sys.argv) >= 3:
        resolution = [int(sys.argv[1]), int(sys.argv[2])]
    else:
        resolution = [300,300]

    data = [[-50, 25, 1],
            [50, 25 , 1],
            [0, 100, 1]]
    
    return data, resolution

def tonormalized(coordinates, resolution):
    for coordinate in (coordinates):
        coordinate[0] = coordinate[0] * 2 / (resolution[0])
        coordinate[1] = coordinate[1] * 2 / (resolution[1])

    return np.array(coordinates, dtype = np.float32)


def transformationFunction(transformationType, transformationData):

    transformationMatrix =  np.identity(4, dtype = np.float32)

    if transformationType == "translation":
        transformationMatrix = np.array([   1.0,0.0,0.0, transformationData[0],
                                            0.0,1.0,0.0, transformationData[1],
                                            0.0,0.0,1.0,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)

    elif transformationType == "rotation":
        cTheta = np.cos(transformationData[0]/180 * math.pi)
        sTheta = np.sin(transformationData[0]/180 * math.pi)

        transformationMatrix = np.array([   cTheta, -sTheta ,0.0,0.0,
                                            sTheta, cTheta,0.0,0.0,
                                            0.0,0.0,1.0,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)
    
    elif transformationType == "scaling":

        transformationMatrix = np.array([   transformationData[0],0.0,0.0,0.0,
                                            0.0,transformationData[1],0.0,0.0,
                                            0.0,0.0,1.0,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)

    elif transformationType == "reflection":
        
        if transformationData[0] == "y":
            transformationMatrix = np.array([   -1.0,0.0,0.0,0.0,
                                                0.0,1.0,0.0,0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)
        elif transformationData[0] == "x":
            transformationMatrix = np.array([   1.0,0.0,0.0,0.0,
                                                0.0,-1.0,0.0,0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)
        elif transformationData[0] == "xy":
            transformationMatrix = np.array([   1.0,1.0,0.0,0.0,
                                                1.0,0.0,0.0,0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)
        else:
            transformationMatrix = np.array([   -1.0,0.0,0.0,0.0,
                                                0.0,-1.0,0.0,0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)
    elif transformationType == "shearing":
        if transformationData[0] == "y":
            transformationMatrix = np.array([   1.0,0.0,0.0,0.0,
                                                transformationData[1],1.0,-transformationData[1] * transformationData[2],0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)
        else:
            transformationMatrix = np.array([   1.0,transformationData[1],-transformationData[1] * transformationData[2],0.0,
                                                0.0,1.0,0.0,0.0,
                                                0.0,0.0,1.0,0.0,
                                                0.0,0.0,0.0,1.0], np.float32)

    return transformationMatrix

    


# initialization function
def initialize():
    global program
    global data
    
    translationMatrix = np.identity(4, np.float32)

    transformationMatrix = transformationFunction("shearing", ["x", 1.2, 0.2])
    # translationMatrix = np.array([   1.0,0.0,0.0, -data[0][0],
    #                                         0.0,1.0,0.0, -data[0][1],
    #                                         0.0,0.0,1.0,0.0,
    #                                         0.0,0.0,0.0,1.0], np.float32)

    print(transformationMatrix)
    print(translationMatrix)
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

    loc = gl.glGetUniformLocation(program, "transformMatrix")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, transformationMatrix)

    loc = gl.glGetUniformLocation(program, "translate")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, translationMatrix)


    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, data.shape[0])
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

data, resolution = buildData()
data = tonormalized(data, resolution)
initialize()

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()