#Tejas D Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)

import math
import sys,pdb,time,copy
from Image import *
import numpy as np
from synthetic_generator import *
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import Image,scipy.misc


try:
	import psyco
	psyco.full()
except ImportError:
	print 'no psyco availiable'


#### Globals #####
surfaces = dict()
cR=0.40; cG=0.40; cB=0.40
WIDTH=300
HEIGHT=300

GENERATE_DATA = False

class BezierPatch():
	def __init__(self):
		# number of patches in x and y direction
		self.nPts = 3
		self.K = 2
		xMin, xMax, yMin, yMax = -0.5, 0.5, -0.5, 0.5
		xStep = (xMax-xMin)/(self.nPts-1)
		yStep = (yMax-yMin)/(self.nPts-1)
		self.divisionsGL = 10

		# initialise a list representing a regular 2D grid of control points.
		# self.controlPoints = [ \
		# 		[ [ yMin+y*yStep, xMin+x*xStep, 0.0 ]  for x in range ( self.nPts )]\
		# 	for y in range( self.nPts ) ]

		self.controlPoints = np.zeros((self.nPts,self.nPts,3))
		for y in range(self.nPts):
			for x in range(self.nPts):
				self.controlPoints[y,x,:]=[ yMin+y*yStep, xMin+x*xStep, 0.0 ]

		# for i in range(self.nPts):
		# 	for j in range(self.nPts):
		# 		print 'P:', self.controlPoints[i][j]
		# 		print 'N:', self.controlPoints2[i,j,:]
		# 		print '----------------------'
		
		# The actual surface is divided into patches of 4 by 4
		# control points
		#self.patch = [ [ [ ] for x in range( 4 )] for y in range( 4 ) ]
		# self.patch = np.zeros((self.psize,self.psize, 3))
		
		# self.patch_color = [ [ [ ] for x in range( 4 )] for y in range( 4 ) ]
		self.patch_color = np.zeros((self.nPts,self.nPts,4))

		#latents
		self.params = {
				#hypers
				'lambda_x':10,
				'lambda_c':10,
				'X_var':0.2,
				'mu_l':-2,
				'mu_u':0,
				'X_l':0,
				'X_u':1,
				'pflip':0.01,
				#latents
				'Z': np.zeros((self.nPts,self.nPts)),
				'theta_c': np.zeros((self.K,3*2)),#rgb
				'C':np.zeros((self.nPts,self.nPts,3)),
				# 'mu_p': np.zeros((self.divisions,self.divisions)),
				'X':np.zeros((self.nPts,self.nPts,3)),
				'mix':np.zeros(self.K),
				'tx':0,
				'ty':0,
				'tz':0,
				'rx':0,
				'ry':0,
				'rz':0,
				'sx':1,
				'sy':1,
				'sz':1
				}
		self.params['X']=copy.deepcopy(self.controlPoints)



	def updateControlPoints(self,pts):
		#self.controlPoints[:,:,2] = np.random.normal(0,1,(self.nPts,self.nPts))
		# self.controlPoints[0:8,0:8,2]-=1
		self.controlPoints = pts

	def updateColors(self,colors):
		self.colors = colors

	def synthetic_render(self):
		glEnable(GL_COLOR_MATERIAL)
		glColor3f(0.7,0.2,0.1)
		#glutSolidCube(1)
		glutSolidCone(0.4,0.7,20,20)
		# glutSolidSphere(0.4,20,20)

	def render(self):
		# plot all surface patches
		# loop over all patches+
		#self.updateControlPoints(0)

		glTranslatef(self.params['tx'],self.params['ty'],self.params['tz'])
		glRotatef(self.params['rx'],1,0,0)
		glRotatef(self.params['ry'],0,1,0)
		glRotatef(self.params['rz'],0,0,1)
		glScalef(self.params['sx'],self.params['sy'],self.params['sz'])
		# self.patch = self.controlPoints[x:x+self.psize,y:y+self.psize,:]
		self.patch_color[:,:,3]=1
		# self.patch_color[:,:,0:3]=np.random.rand(self.nPts,self.nPts,3)#[0.5,0.1,0]
		self.patch_color[:,:,0:3] = self.colors

		glEnable(GL_COLOR_MATERIAL)
		glEnable(GL_MAP2_VERTEX_3);
		glEnable(GL_MAP2_COLOR_4);
		glEnable(GL_AUTO_NORMAL);
		glMap2f( GL_MAP2_VERTEX_3, 0, 1, 0, 1, self.controlPoints )
		glMap2f(GL_MAP2_COLOR_4, 0,1,0,1,self.patch_color)
		glMapGrid2f( self.divisionsGL, 0.0, 1.0, self.divisionsGL, 0.0, 1.0 )
		glEvalMesh2( GL_FILL, 0, self.divisionsGL, 0, self.divisionsGL )



def captureImage(fname='obs.png', save=False):
	global WIDTH,HEIGHT
	glPixelStorei(GL_PACK_ALIGNMENT, 1)
	data = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
	image = Image.fromstring("RGB", (WIDTH, HEIGHT), data)
	image = np.asarray(image)
	if save:	scipy.misc.imsave(fname,image)
	return image


def display(surfaces, capture=False):
	glEnable( GL_LIGHTING )
	glEnable( GL_LIGHT0 )
	glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 0 )
	
	glLightfv( GL_LIGHT0, GL_POSITION, [1, 0.2, -1.5, 1] )
	# xx=2*math.cos(np.random.uniform(0,360))
	# yy=2*math.sin(np.random.uniform(0,360))
	# zz=np.random.uniform(5,6)
	# glLightfv( GL_LIGHT0, GL_POSITION, [xx,yy,zz, 1] )

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


	if GENERATE_DATA:
		glPushMatrix()
		glTranslatef( 0, 0, -3 )
		glRotatef( -60, 0.4, 1, 0)
		surface.synthetic_render()
		glPopMatrix()
		captureImage(save=True)
		glutSwapBuffers( )
	else:
		glPushMatrix()
		glTranslatef( 0, 0, -4 )
		#glRotatef( -45, 0, 1, 0)
		#glRotatef( np.random.uniform(0,360), 0,0,1)
		for ii in range(len(surfaces)):
			surfaces[ii].render()
		glPopMatrix()
		if capture:	
			captured_image = captureImage()
			glutSwapBuffers( )
			return captured_image
		else:
			glutSwapBuffers( )

# get notified of mouse motions
def MouseMotion (x, y):
	global lastx, lasty
	lastx = x
	lasty = y
	glutPostRedisplay ()
	
def keyPressed(self,*args):
	pdb.set_trace()
	# glutLeaveMainLoop()
	# sys.exit()

def init(  ):
	"""Glut init function."""
	glClearColor ( 1, 1, 1, 1 )
	#glClearColor ( 0,0,0, 1 )
	
	glEnable( GL_DEPTH_TEST )
	glShadeModel( GL_SMOOTH )

def setup():
	glutInit( sys.argv )
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
	glutInitWindowSize( WIDTH, HEIGHT )
	glutInitWindowPosition( 0, 0 )
	window = glutCreateWindow( sys.argv[0] )
	init()
	# glutKeyboardFunc(keyPressed)
	#glutDisplayFunc(display)
	# glutIdleFunc( display )
	#glutMainLoop()

if __name__ == "__main__":
	setup()
	surface = BezierPatch()
	for ii in range(100):
		display(surface)

