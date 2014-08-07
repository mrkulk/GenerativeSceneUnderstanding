#Tejas D Kulkarni (tejask@mit.edu | tejasdkulkarni@gmail.com)

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *


def synthetic_data_display():
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
	glMatrixMode( GL_PROJECTION )
	glLoadIdentity( )
	xSize, ySize = glutGet( GLUT_WINDOW_WIDTH ), glutGet( GLUT_WINDOW_HEIGHT )
	gluPerspective(40, float(xSize) / float(ySize), 0.1, 50)
	glMatrixMode( GL_MODELVIEW )
	glLoadIdentity( )
	glTranslatef( 0, 0, -3 )
	cR = 0.3;
	cG = 0.1;
	cB = 0.1;
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, [cR,cG,cB, 1] )
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, [cR,cG,cB, 1] )
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, [cR,cG,cB, 1] )
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50 )
	glEnable( GL_MAP2_VERTEX_3 )

	glutSolidSphere(0.5,40,40)
	glutSwapBuffers( )
	return

# def display():
# 	# glEnable( GL_LIGHTING )
# 	# glEnable( GL_LIGHT0 )
# 	# glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 0 )
	
# 	# glLightfv( GL_LIGHT0, GL_POSITION, [2, 0, 10, 1] )
	
# 	# # xx=np.random.normal(2,1)
# 	# # yy=np.random.normal(0,1)
# 	# # zz=np.random.normal(10,1)
# 	# #glLightfv( GL_LIGHT0, GL_POSITION, [xx,yy,zz, 1] )

# 	# lA = 0.8; glLightfv( GL_LIGHT0, GL_AMBIENT, [lA, lA, lA, 1] )
# 	# lD = 1.0; glLightfv( GL_LIGHT0, GL_DIFFUSE, [lD, lD, lD, 1] )
# 	# lS = 1.0; glLightfv( GL_LIGHT0, GL_SPECULAR, [lS, lS, lS, 1] )
# 	# glEnable( GL_AUTO_NORMAL )

# 	# if False:
# 	# 	data_generator()
# 	# 	return