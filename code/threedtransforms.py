import os
import sys
import ctypes
import numpy as np
import math

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

vertexShaderCode = """
    attribute vec3 position;
    attribute vec4 color;
    varying vec4 vColor; 
    uniform mat4 rotationMatrix[2];
    uniform mat4 transformationMatrix;

    void main(){
        gl_Position = transformationMatrix * rotationMatrix[1] * rotationMatrix[0] * vec4(position, 1.0);
        vColor = color;
    }
    """

fragmentShaderCode = """
    varying vec4 vColor;
    void main(){
        gl_FragColor = vColor;
    }
    """

# -- Building Data -- 
def generateCubeData():
    data = np.zeros(8, [("position", np.float32, 3),
                    ("color",    np.float32, 4)])
        
    data["position"] = (
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
    )

    data['color'] = (
                (0.114, 0.505, 0.345, 1.0),
                (0.483, 0.290, 0.734, 1.0),
                (0.097, 0.513, 0.064, 1.0),
                (0.245, 0.719, 0.592, 1.0),
                (0.583, 0.771, 0.014, 1.0),
                (0.473, 0.211, 0.457, 1.0),
                (0.322, 0.245, 0.574, 1.0),
                (0.083, 0.071, 0.014, 1.0),
                )

    indicesData = np.array([0, 1, 2, 1, 2, 3,
                            4, 5, 6, 5, 6, 7, 
                            4, 2, 5, 2, 5, 1, 
                            5, 1, 3, 5, 3, 7, 
                            0, 4, 2, 4, 2, 6, 
                            2, 6, 3, 6, 3, 7], np.int32)

    return data, indicesData

def generateTransforms(transformationType = None, transformationData = None):
    
    transformationMatrix = np.identity(4, dtype = np.float32)
    transformationMatrix = transformationMatrix.flatten()

    if not transformationType:
        return transformationMatrix
    
    if transformationType == "translation":
        transformationMatrix = np.array([  1.0,0.0,0.0,transformationData[0],
                                            0.0,1.0,0.0,transformationData[1],
                                            0.0,0.0,1.0,transformationData[2],
                                            0.0,0.0,0.0,1.0], np.float32)

    elif transformationType == "rotation":

        cTheta = np.cos(transformationData[1]/180 * math.pi)
        sTheta = np.sin(transformationData[1]/180 * math.pi)

        # x - axis rotation 
        if transformationData[0] == "pitch":
            transformationMatrix = np.array([1.0,0.0,0.0,0.0,
                                            0.0,cTheta,-sTheta,0.0,
                                            0.0,sTheta,cTheta,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)
        
        # y - axis rotation
        elif transformationData[0] == "yaw":
            transformationMatrix = np.array([cTheta,0.0,sTheta,0.0,
                                            0.0,1.0,0.0,0.0,
                                            -sTheta,0.0,cTheta,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)
        
        # z - axis rotation
        elif transformationData[0] == "roll":
            transformationMatrix = np.array([cTheta,-sTheta,0.0,0.0,
                                            sTheta,cTheta,0.0,0.0,
                                            0.0,0.0,1.0,0.0,
                                            0.0,0.0,0.0,1.0], np.float32)

    elif transformationType == "scaling":
        transformationMatrix = np.array([transformationData[0],0.0,0.0,0.0,
                                        0.0,transformationData[1],0.0,0.0,
                                        0.0,0.0,transformationData[2],0.0,
                                        0.0,0.0,0.0,1.0], np.float32)
        
    return transformationMatrix

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
def initialize(transformationMatrix):
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

    degrees = 40
    cTheta = np.cos(degrees/180 * math.pi)
    sTheta = np.sin(degrees/180 * math.pi)
    
    initialRotation = np.array([1.0,0.0,0.0,0.0,
                                0.0,cTheta,-sTheta,0.0,
                                0.0,sTheta,cTheta,0.0,
                                0.0,0.0,0.0,1.0], np.float32)
        
    secondRotation = np.array([cTheta,0.0,sTheta,0.0,
                                0.0,1.0,0.0,0.0,
                                -sTheta,0.0,cTheta,0.0,
                                0.0,0.0,0.0,1.0], np.float32)
    
    rotation = np.array([initialRotation, secondRotation])
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

    loc = gl.glGetUniformLocation(program, "rotationMatrix")
    gl.glUniformMatrix4fv(loc, 2, gl.GL_TRUE, rotation)

    loc = gl.glGetUniformLocation(program, "transformationMatrix")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, transformationMatrix)
    
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

data, indicesData = generateCubeData()

# generate transformation matrix through paramter provision
transformationMatrix = generateTransforms("scaling", [0.8, 0.6, 0.6])

initialize(transformationMatrix = transformationMatrix)

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()




# rot X 30
# -0.5	0.5	    -0.5	0.5	    -0.5	0.5	    -0.5	0.5
# 0.183	0.183	-0.683	-0.683	0.683	0.683	-0.183	-0.183
# 0.683	0.683	0.183	0.183	-0.183	-0.183	-0.683	-0.683
# 1	    1	    1	    1	    1	    1	    1	    1


#rot Y 30
# -0.0915	0.7745	    -0.3415	    0.5245	    -0.5245	    0.3415	    -0.7745	    0.0915
# 0.183	    0.183	    -0.683	    -0.683	    0.683	    0.683	    -0.183	    -0.183
# 0.841478	0.341478	0.408478	-0.091522	0.091522	-0.408478	-0.341478	-0.841478
# 1	1	1	1	1	1	1	1


# (0, 4, 2, 4, 2, 6) : left face
# (2, 6, 3, 6, 3, 7) : bottom face
# (4, 5, 6, 5, 6, 7) : front face
# (0, 1, 2, 1, 2, 3) : back face 
# (4, 2, 5, 2, 5, 1) : top face 
# (5, 1, 3, 5, 3, 7) : right face 


 # data['position'] = (
    #     (-0.04575,   0.0915,    0.420739),
    #     ( 0.38725 ,  0.0915 ,   0.170739),
    #     (-0.17075 , -0.3415,    0.204239),
    #     ( 0.26225,  -0.3415,   -0.045761),
    #     (-0.26225 ,  0.3415,    0.045761),
    #     ( 0.17075,   0.3415,   -0.204239),
    #     (-0.38725 , -0.0915 ,  -0.170739),
    #     ( 0.04575 , -0.0915,   -0.420739)
    #     )