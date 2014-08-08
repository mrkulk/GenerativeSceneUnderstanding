#visualize inference result
#Tejas D Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)
import numpy as np
import pdb,copy,time, pickle
from distributions import *
import bezier,distributions,scipy.stats
import skimage.color,scipy.misc
from skimage import filter
import matplotlib.pyplot as plt
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

surface = None
lastx=0
lasty=0
WIDTH=300
HEIGHT=300
 
# get notified of mouse motions
def MouseMotion (x, y):
	global lastx, lasty 
	lastx = x
	lasty = y
	glutPostRedisplay ()

def display():
	global surface, lasty,lastx
	glEnable( GL_LIGHTING )
	glEnable( GL_LIGHT0 )
	glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 0 )
	
	glLightfv( GL_LIGHT0, GL_POSITION, [1, 0.2, -1.5, 1] )

	lA = 0.4; glLightfv( GL_LIGHT0, GL_AMBIENT, [lA, lA, lA, 1] )
	lD = 1.0; glLightfv( GL_LIGHT0, GL_DIFFUSE, [lD, lD, lD, 1] )
	lS = 1.0; glLightfv( GL_LIGHT0, GL_SPECULAR, [lS, lS, lS, 1] )
	glEnable( GL_AUTO_NORMAL )

	"""OpenGL display function."""
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
	glMatrixMode( GL_PROJECTION )
	glLoadIdentity( )
	xSize, ySize = glutGet( GLUT_WINDOW_WIDTH ), glutGet( GLUT_WINDOW_HEIGHT )
	gluPerspective(40, float(xSize) / float(ySize), 0.1, 50)
	glMatrixMode( GL_MODELVIEW )
	glLoadIdentity( )

	glPushMatrix()
	glTranslatef( 0, 0, -3 )
	#glRotatef( -45, 0, 1, 0)
	#glRotatef( np.random.uniform(0,360), 0,0,1)
	
	glRotatef(lastx, 0.0, 1.0, 0.0);
	glRotatef(lasty, 1.0, 0.0, 0.0);
	surface.render()
	# glEnable(GL_COLOR_MATERIAL)
	# glColor3f(0.5,0.0,0.1)
	# glutSolidCube(1)
	glPopMatrix()
	glutSwapBuffers( )

def init(  ):
	glClearColor ( 1, 1, 1, 1 )
	glEnable( GL_DEPTH_TEST )
	glShadeModel( GL_SMOOTH )

def setup():
	global WIDTH,HEIGHT
	glutInit( sys.argv )
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
	glutInitWindowSize( WIDTH, HEIGHT )
	glutInitWindowPosition( 0, 0 )
	window = glutCreateWindow( sys.argv[0] )
	init()
	glutMotionFunc(MouseMotion)
	glutDisplayFunc(display)
	glutIdleFunc( display )
	glutMainLoop()

if __name__ == "__main__":
	surface = bezier.BezierPatch()
	data = pickle.load(open('params.pkl','rb'))
	surface.updateColors(data['params']['C'])
	surface.updateControlPoints(data['params']['X'])
	surface.divisions = data['divisions']
	surface.nPts = data['nPts']
	setup()

	
