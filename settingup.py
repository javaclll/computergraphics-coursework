import os
import OpenGL.GL as gl 
import OpenGL.GLUT as glut

def display():
    # Clearing the window buffer for display
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    glut.glutSwapBuffers()

def reshape(width, height):
    gl.glViewport(800, 800, width, height)
    print(f"Resolution of the Window : {width} x {height}")

# function to exit the Graphics Window on escape 
def keyboard( key, x, y ):
    if key == b'\x1b':
        os._exit(1)

#Initializing GLUT for Window Rendering
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutCreateWindow("Graphics Window")

#Window Reshaping
glut.glutReshapeWindow(1920, 1080)

#Resolution of the Display System
print(f"Resolution of the Display System : {glut.glutGet(glut.GLUT_SCREEN_WIDTH)} x {glut.glutGet(glut.GLUT_SCREEN_HEIGHT)}")

glut.glutReshapeFunc(reshape)
glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)
glut.glutMainLoop()